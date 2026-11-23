#include "kernel_operator.h"
#include "lib/matmul_intf.h"  // Required by the generated MIX entry.
#include "clef_mlp_gate_up_pair_tiling.h"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"

namespace {
using namespace AscendC;
using namespace Catlass;
using namespace ClefGateUp;
using ND = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
using NZ = Gemm::GemmType<bfloat16_t, layout::nZ>;
using Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

constexpr uint16_t Ready = 0;
constexpr uint16_t Recycled = 1;
constexpr uint32_t ChunkRows = 32;

template<uint32_t Rows, uint32_t Columns>
__aicore__ inline uint32_t Slot(uint32_t group, uint32_t ordinal)
{
    return (group * RingSlots + ordinal % RingSlots) * Rows * Columns;
}

__aicore__ inline bool BatchEnd(uint32_t ordinal, uint32_t task, uint32_t tasks)
{
    return (ordinal + 1) % RingSlots == 0 || task + Groups >= tasks;
}

template<uint32_t Rows, uint32_t Columns, bool FusedOutput>
__aicore__ inline void RunCube(GM_ADDR x, GM_ADDR weight, GM_ADDR out, GM_ADDR workspace,
                               uint32_t tokens)
{
    using Mmad = Gemm::Block::BlockMmad<Gemm::MmadAtlasA2Preload<true, false>,
        GemmShape<Rows, Columns, 256>, GemmShape<Rows, Columns, 64>, ND, NZ, ND>;
    static_assert(std::is_same_v<typename Mmad::ElementAccumulator, float>);
    static_assert(Rows * Columns * sizeof(float) == 128 * 1024);

    Arch::Resource<Arch::AtlasA2> resources;
    Mmad gemm(resources);
    GlobalTensor<bfloat16_t> input, paired, output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
    paired.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(weight));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(FusedOutput ? workspace : out));

    const layout::RowMajor inputLayout(tokens, Hidden);
    const auto weightLayout = layout::nZ::MakeLayout<bfloat16_t>(Hidden, 2 * Intermediate);
    const layout::RowMajor outputLayout(FusedOutput ? Rows : tokens,
                                        FusedOutput ? Columns : 2 * Intermediate);
    Scheduler scheduler(GemmCoord(tokens, 2 * Intermediate, Hidden), MatrixCoord(Rows, Columns));
    const auto group = GetBlockIdx();
    const auto tasks = scheduler.GetCoreLoops();
    for (uint32_t task = group, ordinal = 0; task < tasks; task += Groups, ++ordinal) {
        const auto tile = scheduler.GetBlockCoord(task);
        const auto shape = scheduler.GetActualBlockShape(tile);
        const auto a = input[tile.m() * Rows * Hidden];
        const auto b = paired[weightLayout.GetOffset(MatrixCoord(0U, tile.n() * Columns))];
        const auto offset = FusedOutput ? Slot<Rows, Columns>(group, ordinal)
                                        : tile.m() * Rows * 2 * Intermediate + tile.n() * Columns;
        gemm(a, inputLayout, b, weightLayout, output[offset], outputLayout,
             a, b, shape, shape, true, false);
        if constexpr (FusedOutput) {
            // The corresponding vector lanes read the tile only after Fixpipe publishes it.
            CrossCoreSetFlag<0x2, PIPE_FIX>(Ready);
            if (BatchEnd(ordinal, task, tasks)) {
                CrossCoreWaitFlag(Recycled);
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
}

template<uint32_t Rows, uint32_t Columns>
__aicore__ inline void RunVector(GM_ADDR out, GM_ADDR workspace, uint32_t tokens)
{
    constexpr uint32_t LaneRows = Rows / 2;
    constexpr uint32_t Half = Columns / 2;
    constexpr uint32_t Elements = ChunkRows * Half;
    constexpr uint16_t CopyWidth = Half * sizeof(bfloat16_t) / 32;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input;
    TQue<QuePosition::VECOUT, 1> output;
    TBuf<TPosition::VECCALC> gateBuffer, upBuffer, activationBuffer;
    pipe.InitBuffer(input, 1, 2 * Elements * sizeof(bfloat16_t));
    pipe.InitBuffer(output, 1, Elements * sizeof(bfloat16_t));
    pipe.InitBuffer(gateBuffer, Elements * sizeof(float));
    pipe.InitBuffer(upBuffer, Elements * sizeof(float));
    pipe.InitBuffer(activationBuffer, Elements * sizeof(float));

    GlobalTensor<bfloat16_t> ring, activation;
    ring.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspace));
    activation.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out));

    Scheduler scheduler(GemmCoord(tokens, 2 * Intermediate, Hidden), MatrixCoord(Rows, Columns));
    const auto group = GetBlockIdx() / GetSubBlockNum();
    const auto laneBegin = GetSubBlockIdx() * LaneRows;
    const auto tasks = scheduler.GetCoreLoops();
    for (uint32_t task = group, ordinal = 0; task < tasks; task += Groups, ++ordinal) {
        const auto tile = scheduler.GetBlockCoord(task);
        const auto rows = scheduler.GetActualBlockShape(tile).m();
        const auto end = rows < laneBegin + LaneRows ? rows : laneBegin + LaneRows;
        CrossCoreWaitFlag(Ready);
        for (uint32_t row = laneBegin; row < end; row += ChunkRows) {
            const auto valid = end - row < ChunkRows ? end - row : ChunkRows;
            const auto source = Slot<Rows, Columns>(group, ordinal) + row * Columns;
            const auto dest = (tile.m() * Rows + row) * Intermediate + tile.n() * Half;
            const DataCopyParams read{static_cast<uint16_t>(valid), CopyWidth, CopyWidth, 0};
            auto halves = input.AllocTensor<bfloat16_t>();
            DataCopy(halves, ring[source], read);
            DataCopy(halves[Elements], ring[source + Half], read);
            input.EnQue(halves);
            halves = input.DeQue<bfloat16_t>();

            auto gate = gateBuffer.Get<float>();
            auto up = upBuffer.Get<float>();
            auto value = activationBuffer.Get<float>();
            const auto count = valid * Half;
            Cast(gate, halves, RoundMode::CAST_NONE, count);
            Cast(up, halves[Elements], RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            input.FreeTensor(halves);
            Silu(value, gate, count);
            PipeBarrier<PIPE_V>();
            Mul(value, value, up, count);
            PipeBarrier<PIPE_V>();

            auto bf16 = output.AllocTensor<bfloat16_t>();
            Cast(bf16, value, RoundMode::CAST_RINT, count);
            output.EnQue(bf16);
            bf16 = output.DeQue<bfloat16_t>();
            const DataCopyParams write{static_cast<uint16_t>(valid), CopyWidth, 0,
                                       (Intermediate - Half) * sizeof(bfloat16_t) / 32};
            DataCopy(activation[dest], bf16, write);
            output.FreeTensor(bf16);
        }
        // Both lanes acknowledge every ring batch, including an empty tail lane.
        if (BatchEnd(ordinal, task, tasks)) {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(Recycled);
        }
    }
    PipeBarrier<PIPE_ALL>();
}
}  // namespace

extern "C" __global__ __aicore__ void clef_mlp_gate_up_pair(
    GM_ADDR x, GM_ADDR pairedWeight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(ClefGateUp::TilingData);
    GET_TILING_DATA(params, tiling);
    const auto userWorkspace = AscendC::GetUserWorkspace(workspace);
    if ASCEND_IS_AIC {
        switch (params.mode) {
            case Raw:
                RunCube<128, 256, false>(x, pairedWeight, y, userWorkspace, params.tokens);
                break;
            case Fused:
                RunCube<128, 256, true>(x, pairedWeight, y, userWorkspace, params.tokens);
                break;
            case ShortRaw:
                RunCube<256, 128, false>(x, pairedWeight, y, userWorkspace, params.tokens);
                break;
            case ShortFused:
                RunCube<256, 128, true>(x, pairedWeight, y, userWorkspace, params.tokens);
                break;
        }
    } else if ASCEND_IS_AIV {
        if (params.mode == Fused) {
            RunVector<128, 256>(y, userWorkspace, params.tokens);
        } else if (params.mode == ShortFused) {
            RunVector<256, 128>(y, userWorkspace, params.tokens);
        }
    }
}
