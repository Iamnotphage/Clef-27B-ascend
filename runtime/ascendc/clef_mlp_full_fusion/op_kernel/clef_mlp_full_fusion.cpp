#include "kernel_operator.h"
#include "lib/matmul_intf.h"  // Required by CANN's generated MIX entry.
#include "clef_mlp_full_fusion_tiling.h"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"

namespace {
using namespace AscendC;
using namespace Catlass;
using namespace ClefMlp;
using ND = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
using NZ = Gemm::GemmType<bfloat16_t, layout::nZ>;
using Mmad = Gemm::Block::BlockMmad<Gemm::MmadAtlasA2Preload<true, false>,
    GemmShape<TileM, TileN, 256>, GemmShape<TileM, TileN, 64>, ND, NZ, ND>;
using Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
static_assert(std::is_same_v<typename Mmad::ElementAccumulator, float>);

constexpr uint16_t ReadyFlag = 0;  // Two consecutive publications per batch.
constexpr uint16_t RecycledFlag = 1;
constexpr uint16_t ActivationFlag = 4;
constexpr uint32_t LaneRows = TileM / 2;
constexpr uint32_t ChunkRows = 32;
constexpr uint32_t ChunkElements = ChunkRows * TileN;
constexpr uint16_t CopyWidth = TileN * sizeof(bfloat16_t) / 32;

__aicore__ inline uint32_t SlotOffset(uint32_t group, uint32_t ordinal)
{
    return (group * RingSlots + ordinal % RingSlots) * SlotElements;
}

__aicore__ inline void RunCube(GM_ADDR x, GM_ADDR gateWeight, GM_ADDR downWeight,
                              GM_ADDR output, GM_ADDR userWorkspace, uint32_t tokens)
{
    Arch::Resource<Arch::AtlasA2> resources;
    Mmad gemm(resources);
    GlobalTensor<bfloat16_t> input, gate, down, ring, activation, y;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
    gate.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gateWeight));
    down.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(downWeight));
    ring.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(userWorkspace));
    activation = ring[RingElements];
    y.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));

    const layout::RowMajor inputLayout(tokens, H);
    const auto gateLayout = layout::nZ::MakeLayout<bfloat16_t>(H, 2 * I);
    const layout::RowMajor ringLayout(TileM, SlotRow);
    Scheduler gates(GemmCoord(tokens, I, H), MatrixCoord(TileM, TileN));
    const auto group = GetBlockIdx();
    for (uint32_t task = group, ordinal = 0; task < gates.GetCoreLoops();
         task += Groups, ++ordinal) {
        const auto tile = gates.GetBlockCoord(task);
        const auto shape = gates.GetActualBlockShape(tile);
        const auto a = input[tile.m() * TileM * H];
        const auto bGate = gate[gateLayout.GetOffset(MatrixCoord(0U, tile.n() * TileN))];
        const auto bUp = gate[gateLayout.GetOffset(MatrixCoord(0U, I + tile.n() * TileN))];
        const auto slot = SlotOffset(group, ordinal);
        gemm(a, inputLayout, bGate, gateLayout, ring[slot], ringLayout,
             a, bGate, shape, shape, true, false);
        gemm(a, inputLayout, bUp, gateLayout, ring[slot + TileN], ringLayout,
             a, bUp, shape, shape, true, false);
        // Inline the fixed CATLASS reverse-depth protocol so both flag
        // endpoints and the batch lifetime are explicit in this kernel.
        CrossCoreSetFlag<0x2, PIPE_FIX>(ReadyFlag);
        if ((ordinal + 1) % RingSlots == 0) {
            CrossCoreWaitFlag(RecycledFlag);
        }
    }

    // All AIVs publish their writes via the documented global AIV barrier.
    // The subsequent peer flag covers both the partly filled final ring batch
    // and the activation fanout from every group to each complete down tile.
    CrossCoreWaitFlag(ActivationFlag);
    const layout::RowMajor activationLayout(tokens, I);
    const auto downLayout = layout::nZ::MakeLayout<bfloat16_t>(I, H);
    const layout::RowMajor outputLayout(tokens, H);
    Scheduler outputs(GemmCoord(tokens, H, I), MatrixCoord(TileM, TileN));
    for (uint32_t task = group; task < outputs.GetCoreLoops(); task += Groups) {
        const auto tile = outputs.GetBlockCoord(task);
        const auto shape = outputs.GetActualBlockShape(tile);
        const auto a = activation[tile.m() * TileM * I];
        const auto b = down[downLayout.GetOffset(MatrixCoord(0U, tile.n() * TileN))];
        const auto out = y[tile.m() * TileM * H + tile.n() * TileN];
        gemm(a, activationLayout, b, downLayout, out, outputLayout,
             a, b, shape, shape, true, false);
    }
    PipeBarrier<PIPE_ALL>();
}

class VectorEpilogue {
public:
    __aicore__ inline void Init(TPipe& pipe)
    {
        pipe.InitBuffer(input_, 1, 2 * ChunkElements * sizeof(bfloat16_t));
        pipe.InitBuffer(output_, 1, ChunkElements * sizeof(bfloat16_t));
        pipe.InitBuffer(gate_, ChunkElements * sizeof(float));
        pipe.InitBuffer(up_, ChunkElements * sizeof(float));
        pipe.InitBuffer(silu_, ChunkElements * sizeof(float));
    }

    __aicore__ inline void Consume(const GlobalTensor<bfloat16_t>& slot,
                                  const GlobalTensor<bfloat16_t>& activation,
                                  const GlobalTensor<bfloat16_t>& diagnostics,
                                  uint32_t rows, uint32_t tokens, bool capture)
    {
        const auto count = rows * TileN;
        auto input = input_.AllocTensor<bfloat16_t>();
        const DataCopyParams inCopy{static_cast<uint16_t>(rows), CopyWidth, CopyWidth, 0};
        DataCopy(input, slot, inCopy);
        DataCopy(input[ChunkElements], slot[TileN], inCopy);
        input_.EnQue(input);
        input = input_.DeQue<bfloat16_t>();
        const auto gateFloat = gate_.Get<float>();
        const auto upFloat = up_.Get<float>();
        const auto silu = silu_.Get<float>();
        Cast(gateFloat, input, RoundMode::CAST_NONE, count);
        Cast(upFloat, input[ChunkElements], RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        input_.FreeTensor(input);

        if (capture) {
            // Exact BF16 round trip for finite inputs, using the same GM tile
            // that the epilogue consumed. Diagnostic copies are out of timing.
            Store(gateFloat, diagnostics, rows);
            Store(upFloat, diagnostics[tokens * I], rows);
        }

        Silu(silu, gateFloat, count);
        PipeBarrier<PIPE_V>();
        Mul(silu, silu, upFloat, count);
        PipeBarrier<PIPE_V>();
        Store(silu, activation, rows);
        if (capture) {
            Store(silu, diagnostics[2 * tokens * I], rows);
        }
    }

private:
    __aicore__ inline void Store(const LocalTensor<float>& values,
                                const GlobalTensor<bfloat16_t>& dst, uint32_t rows)
    {
        auto value = output_.AllocTensor<bfloat16_t>();
        Cast(value, values, RoundMode::CAST_RINT, rows * TileN);
        output_.EnQue(value);
        value = output_.DeQue<bfloat16_t>();
        const DataCopyParams outCopy{static_cast<uint16_t>(rows), CopyWidth, 0,
                                     (I - TileN) * sizeof(bfloat16_t) / 32};
        DataCopy(dst, value, outCopy);
        output_.FreeTensor(value);
    }

    TQue<QuePosition::VECIN, 1> input_;
    TQue<QuePosition::VECOUT, 1> output_;
    TBuf<TPosition::VECCALC> gate_, up_, silu_;
};

__aicore__ inline void RunVector(GM_ADDR output, GM_ADDR userWorkspace,
                                uint32_t tokens, bool capture)
{
    TPipe pipe;
    VectorEpilogue epilogue;
    epilogue.Init(pipe);
    GlobalTensor<bfloat16_t> ring, activation, diagnostics;
    ring.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(userWorkspace));
    activation = ring[RingElements];
    if (capture) {
        diagnostics.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output) + tokens * H);
    } else {
        // Never accessed in the measurement branch.
        diagnostics = activation;
    }
    const auto group = GetBlockIdx() / GetSubBlockNum();
    const auto laneBegin = GetSubBlockIdx() * LaneRows;
    Scheduler gates(GemmCoord(tokens, I, H), MatrixCoord(TileM, TileN));
    uint32_t ordinal = 0;
    for (uint32_t task = group; task < gates.GetCoreLoops(); task += Groups, ++ordinal) {
        const auto tile = gates.GetBlockCoord(task);
        const auto rows = gates.GetActualBlockShape(tile).m();
        const auto end = rows < laneBegin + LaneRows ? rows : laneBegin + LaneRows;
        CrossCoreWaitFlag(ReadyFlag);
        for (uint32_t row = laneBegin; row < end; row += ChunkRows) {
            const auto valid = end - row < ChunkRows ? end - row : ChunkRows;
            const auto source = SlotOffset(group, ordinal) + row * SlotRow;
            const auto dest = (tile.m() * TileM + row) * I + tile.n() * TileN;
            epilogue.Consume(ring[source], activation[dest], diagnostics[dest],
                             valid, tokens, capture);
        }
        // Both lanes participate even if the M tail leaves one lane empty.
        if ((ordinal + 1) % RingSlots == 0) {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(RecycledFlag);
        }
    }
    Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
    CrossCoreSetFlag<0x2, PIPE_MTE3>(ActivationFlag);
    PipeBarrier<PIPE_ALL>();
}
}  // namespace

extern "C" __global__ __aicore__ void clef_mlp_full_fusion(
    GM_ADDR x, GM_ADDR gateWeight, GM_ADDR downWeight, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(ClefMlp::TilingData);
    GET_TILING_DATA(params, tiling);
    const auto userWorkspace = AscendC::GetUserWorkspace(workspace);
    if ASCEND_IS_AIC {
        RunCube(x, gateWeight, downWeight, output, userWorkspace, params.tokens);
    }
    if ASCEND_IS_AIV {
        RunVector(output, userWorkspace, params.tokens, params.diagnostic != 0);
    }
}
