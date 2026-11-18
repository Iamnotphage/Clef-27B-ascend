#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "adv_api/hccl/hccl.h"
#include "clef_matmul_ar_add_rms_norm_tiling.h"

using namespace AscendC;

// Group E, first version: one operator containing
//   MatMul (X[M,K] x W[K,N] -> partial[M,N])
//   -> TP AllReduce (partial -> reduced, out-of-place)
//   -> full-N Add + RMSNorm (z = reduced + residual; out = z*rstd*gamma).
//
// The AllReduce and normalisation half is a direct reuse of the already
// verified ClefArAddRmsNorm contract:
//   * InitV2(GetHcclContext<HCCL_GROUP_ID_0>(), &tiling->mc2InitTiling);
//   * SetCcTilingV2(offsetof(tiling, mc2CcTiling)) -- offset relative to the
//     initTiling pointer, which is why mc2InitTiling is the first member;
//   * AllReduce<true>(send, recv, m*n, BFP16, SUM, repeat) is out-of-place;
//   * after Wait() the reduced buffer is read only after a full-cache
//     invalidate;
//   * only core 0 writes hcclMsgArea, the other cores wait on a GM flag.
//
// Device-qualified facts this file is built around:
//   * mm_enable=0 is clean, so the vector path is sound;
//   * the scalar-access fault reproduces at Init/SetOrgShape time (mm_mode=1)
//     and with a single core (mm_dim=1), so it is not the iterate and not the
//     multi-core split;
//   * use_hccl=0 hangs, use_hccl=2 (lifecycle only) is clean -- 2 is the
//     no-AllReduce control arm.
class ClefMatmulArAddRmsNormKernel {
public:
    using AT = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
    // Real contract: the service keeps the down-projection weight in
    // FRACTAL_NZ (CLEF_MLP_BF16_NZ does one npu_format_cast(..., 29) at load
    // time), so the operator assumes "weight arrives already NZ" and never
    // converts.  The ND variant remains at commit b1f1741 for the numeric
    // gate; host SetBType and SetTensorB are driven by the same attribute.
    using BT = MatmulType<TPosition::GM, CubeFormat::NZ, bfloat16_t, false>;
    using CT = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
    using BiasT = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;

    __aicore__ inline void SetTiling(ClefMatmulArAddRmsNormTilingData* t) { tiling_ = t; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR residual, GM_ADDR gamma,
                                GM_ADDR out, GM_ADDR residualOut, GM_ADDR rstd,
                                GM_ADDR diag, GM_ADDR workspace,
                                ClefMatmulArAddRmsNormTilingData* t) {
        m_ = t->m;
        n_ = t->n;
        k_ = t->k;
        eps_ = t->eps;
        invN_ = t->invN;
        mPad_ = t->mPad;
        repeat_ = static_cast<uint8_t>(t->repeat);
        hcclMode_ = t->useHccl;
        wsGm_ = workspace;

        xGm_ = x;
        wGm_ = w;

        const uint64_t mnBytes = static_cast<uint64_t>(m_) * n_ * 2;
        // The KFC system region lives in front of the user region; only the
        // user part may be used for the matmul output, the reduce output and
        // the completion flag.
        GM_ADDR userWs = AscendC::GetUserWorkspace(workspace);
        wsGm_ = userWs;
        partialGm_ = userWs;
        arGm_ = (userWs != nullptr) ? (GM_ADDR)((__gm__ uint8_t*)userWs + mnBytes) : nullptr;
        if (userWs != nullptr) {
            flagGm_ = (__gm__ float*)((__gm__ uint8_t*)userWs + 2 * mnBytes);
            flagG_.SetGlobalBuffer(flagGm_, 8);
        }
        diag_.SetGlobalBuffer((__gm__ float*)diag, 8);

        residualGm_ = residual;
        r_.SetGlobalBuffer((__gm__ bfloat16_t*)residual, static_cast<uint64_t>(m_) * n_);
        g_.SetGlobalBuffer((__gm__ bfloat16_t*)gamma, n_);
        o_.SetGlobalBuffer((__gm__ bfloat16_t*)out, static_cast<uint64_t>(m_) * n_);
        ro_.SetGlobalBuffer((__gm__ bfloat16_t*)residualOut, static_cast<uint64_t>(m_) * n_);
        rs_.SetGlobalBuffer((__gm__ float*)rstd, m_);

        // Only the tiny diag buffer is taken before the cube stage, so the
        // matmul's own InitBuffer cannot collide with the ~120 KB the
        // normalisation pass needs when norm_ub_late is set.
        pipe_.InitBuffer(diagT_, 32);
        if (t->normLate == 0u) {
            InitNormBuffers();
        }
    }

    __aicore__ inline void InitNormBuffers() {
        if (normBuffersReady_) return;
        normBuffersReady_ = true;
        pipe_.InitBuffer(bx_, n_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(br_, n_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(bg_, n_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(bo_, n_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(bout_, n_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(bz_, n_ * sizeof(float));
        pipe_.InitBuffer(bt_, n_ * sizeof(float));
        pipe_.InitBuffer(bw_, n_ * sizeof(float));
        pipe_.InitBuffer(bs_, 64 * sizeof(float));
        pipe_.InitBuffer(bwr_, 8 * 1024);
        pipe_.InitBuffer(bgst_, 2 * 1024);
    }

    __aicore__ inline void WriteTilingDiag() {
        if (GetBlockIdx() != 0) return;
        LocalTensor<float> d = diagT_.Get<float>();
        // Output ABI is unchanged: diag stays 8 floats.
        // Default (mc_diag = 0): 5 = usedCoreNum, 6 = baseM, 7 = depthA1.
        // With mc_diag = 1 the three slots are repurposed for the multi-core
        // contract, because the caller's 8-float buffer has no spare slot:
        //   5 = workers (tiling's usedCoreNum, the worker count)
        //   6 = singleCoreM (per-worker M extent used for the slice)
        //   7 = GetBlockNum() (actual launch blocks)
        // The requested budget is not in diag; it stays a host-side record
        // (tiling mmDim), so budget / tiling-answer / launch are never conflated.
        if (tiling_->mcDiag != 0u) {
            d.SetValue(5, static_cast<float>(tiling_->cubeTiling.usedCoreNum));
            d.SetValue(6, static_cast<float>(tiling_->cubeTiling.singleCoreM));
            d.SetValue(7, static_cast<float>(GetBlockNum()));
        } else {
            d.SetValue(5, static_cast<float>(tiling_->cubeTiling.usedCoreNum));
            d.SetValue(6, static_cast<float>(tiling_->cubeTiling.baseM));
            d.SetValue(7, static_cast<float>(tiling_->cubeTiling.depthA1));
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(diag_[0], d, 8);
        PipeBarrier<PIPE_ALL>();
    }

    // Records which cube stage just completed (0 = none) plus the tiling
    // fields, then the caller returns immediately.  Only block 0 writes, and
    // every block leaves together so no barrier is left half-entered.
    __aicore__ inline void WriteStageDiag(uint32_t stageDone) {
        if (GetBlockIdx() != 0) return;
        LocalTensor<float> d = diagT_.Get<float>();
        float marker = 0.0f;
        if (stageDone == 1u) marker = 1.0f;
        else if (stageDone == 2u) marker = 2.0f;
        else if (stageDone == 3u) marker = 3.0f;
        else if (stageDone == 4u) marker = 4.0f;
        else if (stageDone == 5u) marker = 5.0f;
        d.SetValue(0, marker);
        d.SetValue(5, static_cast<float>(tiling_->cubeTiling.usedCoreNum));
        d.SetValue(6, static_cast<float>(tiling_->cubeTiling.baseM));
        d.SetValue(7, static_cast<float>(tiling_->cubeTiling.depthA1));
        PipeBarrier<PIPE_ALL>();
        DataCopy(diag_[0], d, 8);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process() {
        int32_t mmRc = 0;
        int32_t arRc = 0;
        int32_t rankId = -1;
        int32_t rankDim = -1;

        WriteTilingDiag();

        // ---- 1) MatMul X[M,K] x W[K,N] -> partial[M,N] (workspace region 0) --
        // mm_enable=0 skips the whole cube stage.  Only blocks below the
        // tiling's usedCoreNum may enter the matmul, so SetDim and SetBlockDim
        // can be decoupled safely.
        // The KFC client/server pair must be complete on every participating
        // block: each AIV client's destructor posts a service-quit message and
        // the per-AIC server only leaves its loop when it has collected
        // MIX_NUM of them (kernel_kfc.h isRun() -> quitSize < MIX_NUM).  Gating
        // the registration on usedCoreNum leaves servers short of quits and
        // they spin forever, which surfaces as "aicore execution times out".
        // So only the compute calls are gated; the wiring scope encloses them
        // so the client outlives every matmul call.
        if (tiling_->mmEnable != 0) {
            GlobalTensor<bfloat16_t> aG, bG, cG;
            aG.SetGlobalBuffer((__gm__ bfloat16_t*)xGm_, static_cast<uint64_t>(m_) * k_);
            bG.SetGlobalBuffer((__gm__ bfloat16_t*)wGm_, static_cast<uint64_t>(k_) * n_);
            cG.SetGlobalBuffer((__gm__ bfloat16_t*)partialGm_, static_cast<uint64_t>(m_) * n_);
            // mm_stage runs up to a chosen cube call, records it in diag and
            // returns immediately, so the last diag value written shows which
            // call completed before the fault:
            //   1 = Init            2 = + SetOrgShape    3 = + SetTensorA
            //   4 = + SetTensorB    5 = + IterateAll
            // mm_stage=0 keeps the normal path (mm_mode still selects whether
            // IterateAll runs).
            const uint32_t stage = tiling_->mmStage;
            // Official wiring.  This kernel is compiled for both the cube and
            // the vector pass: on the cube pass REGIST_MATMUL_OBJ runs the KFC
            // server loop and returns, on the vector pass it creates the
            // KfcCommClient and wires it before the object is initialised.
            // Without it the client is uninitialised and Init() walks
            // server-owned internal buffers -- the device fault we saw.
            REGIST_MATMUL_OBJ(&pipe_, GetSysWorkSpacePtr(), mm_,
                              &tiling_->cubeTiling);
            // Only the compute calls are gated; the wiring above runs on every
            // block so each per-AIC server collects its MIX_NUM quits.
            // ---- per-worker M slice ------------------------------------------
            // A single vector-side client per worker owns a distinct M range.
            // N and K are NOT split, so the NZ weight pointer needs no offset at
            // all: the production NZ offset helper is tailN * AlignUp<16>(k),
            // which is 0 for N block index 0
            // (vllm-ascend gmm grouped_matmul_swiglu_quant.h:222-231; its
            // caller offsets only A and C for an M split at :169-172 and adds
            // GetWOffset(tailN, k) only for an N split at :174).
            // We deliberately do NOT rely on any AIV->AIC index mapping: each
            // participating block owns a distinct M range, so the result is
            // correct whichever server consumes the message.
            const uint32_t workers = (tiling_->cubeTiling.usedCoreNum == 0u)
                                         ? 1u
                                         : static_cast<uint32_t>(tiling_->cubeTiling.usedCoreNum);
            const uint32_t myBlock = static_cast<uint32_t>(GetBlockIdx());
            uint32_t singleM = (tiling_->cubeTiling.singleCoreM == 0u)
                                         ? ((m_ + workers - 1u) / workers)
                                         : static_cast<uint32_t>(tiling_->cubeTiling.singleCoreM);
            // Coverage guard: the tiling's singleCoreM is used only when it
            // still tiling-covers the whole M with `workers` workers.  Trusting
            // it blindly would leave rows uncovered whenever it is smaller than
            // ceil(m / workers) (a pure-CPU sweep found exactly that case), and
            // an uncovered row is an unwritten row in `partial`.
            if (singleM == 0u || singleM * workers < m_) {
                singleM = (m_ + workers - 1u) / workers;
            }
            const uint32_t startM = myBlock * singleM;
            const uint32_t curSingleM =
                (startM >= m_) ? 0u : (((m_ - startM) < singleM) ? (m_ - startM) : singleM);
            const bool isWorker = (myBlock < workers) && (singleM > 0u) && (curSingleM > 0u);
            if (isWorker) {
                if (stage == 1u) { WriteStageDiag(1u); }  // after wiring/Init
                if (stage != 1u) {
                    mm_.SetOrgShape(m_, n_, k_);
                    if (stage == 2u) { WriteStageDiag(2u); }
                }
                if (stage != 1u && stage != 2u) {
                    // A is ND [M,K]; this worker owns rows [startM, startM+curSingleM)
                    mm_.SetTensorA(aG[startM * k_], false);
                    if (stage == 3u) { WriteStageDiag(3u); }
                }
                if (stage != 1u && stage != 2u && stage != 3u) {
                    // mm_btrans=1 tells the cube B is stored column-major; host
                    // tiling and the kernel share the same attribute.  B itself
                    // is never offset here (N and K are not split).
                    mm_.SetTensorB(bG, tiling_->mmBTrans != 0u);
                    if (stage == 4u) { WriteStageDiag(4u); }
                }
                if (stage == 0u || stage == 5u) {
                    if (tiling_->mmMode >= 2u) {
                        // Per-worker shape and C offset.  The tail worker passes
                        // its smaller curSingleM; SetSingleShape is the documented
                        // per-core shape entry (it forwards to SetTail, which sets
                        // mIter_/nIter_ -- matmul_client.h:234-259).
                        mm_.SetSingleShape(curSingleM, n_, k_);
                        // waitIterateAll=true so End() retires a tile that has
                        // actually been produced.
                        mm_.IterateAll(cG[startM * n_], 0, false, true);
                    }
                    if (stage == 5u) { WriteStageDiag(5u); }
                    // The documented call order ends with End(); on the client
                    // path it posts MMFUN_END, which retires the pending cube
                    // task before the client's scope ends.
                    mm_.End();
                }
                mmRc = 0;
            }
        }
        if (tiling_->mmStage >= 1u && tiling_->mmStage <= 5u) {
            // Stage-gated run: leave together, no barrier, no AR, no norm.
            return;
        }
        // ---- cross-worker visibility for the AR / Norm consumers --------------
        // Each worker wrote only its own M rows of `partial`; the AR publisher
        // (block 0) must not read before every worker's rows are there.  What
        // the barrier is allowed to rest on, with line numbers:
        //
        // 1. Every AIV block reaches this line.  The worker guard above only
        //    skips the matmul calls, so the barrier set is the full AIV set --
        //    the same set on entry and exit.
        // 2. Client-side completion is explicit: IterateAll is called with
        //    waitIterateAll=true, whose wait entry is matmul_client.h:772
        //    (WaitIterateAll), and End() posts MMFUN_END (matmul_client.h:695-701)
        //    to retire the cube task before this point.
        // 3. The server side only leaves its loop after collecting MIX_NUM
        //    quits (kernel_kfc.h:60-63, `isRun() { return quitSize < MIX_NUM; }`,
        //    with the quit counting at :123-128 and :199-218), so the wiring
        //    scope above keeps the pair complete.
        //
        // SyncAll<true> is the isAIVOnly form; the template parameter is declared
        // at basic_api/kernel_operator_block_sync_intf.h:77-87 (`template <bool
        // isAIVOnly = true>`).  The default value alone is NOT a usability proof,
        // so what remains unproven and must be checked on device:
        //   (a) that this AIV-only barrier orders the fixpipe/GM writes of a
        //       worker's cube stage against another AIV block's GM reads -- the
        //       primitives' memory-ordering contract is not stated in the header;
        //   (b) that the barrier's participating set equals the set of blocks
        //       that actually issued matmul work when usedCoreNum exceeds the
        //       number of AIC cores;
        //   (c) that no AIC block is still required to participate at this point
        //       (the earlier "AIC already exited, so a barrier times out" comment
        //       was based on a hypothesis that was later retracted, so it is no
        //       longer offered as a reason either way).
        SyncAll<true>();
        PipeBarrier<PIPE_ALL>();

        if (tiling_->normLate != 0u) {
            InitNormBuffers();
        }

        // ---- 2) TP AllReduce partial -> reduced (out-of-place) ----------------
        if (hcclMode_ >= 1u && arGm_ != nullptr) {
            __gm__ float* flagPtr = flagGm_;
            if (GetBlockIdx() == 0) {
                Hccl hccl;
                hccl.InitV2(AscendC::GetHcclContext<HCCL_GROUP_ID_0>(), &tiling_->mc2InitTiling);
                const int32_t setRc = hccl.SetCcTilingV2(
                    offsetof(ClefMatmulArAddRmsNormTilingData, mc2CcTiling));
                rankId = static_cast<int32_t>(hccl.GetRankId());
                rankDim = static_cast<int32_t>(hccl.GetRankDim());
                if (hcclMode_ == 1u) {
                    HcclHandle h = hccl.AllReduce<true>(partialGm_, arGm_,
                                                        static_cast<uint64_t>(m_) * n_,
                                                        HCCL_DATA_TYPE_BFP16, HCCL_REDUCE_SUM,
                                                        repeat_);
                    PipeBarrier<PIPE_ALL>();
                    arRc = hccl.Wait(h) * 100 + setRc;
                } else {
                    // Lifecycle-only arm: prove InitV2/Finalize pairing without
                    // publishing any collective.
                    arRc = setRc;
                }
                hccl.Finalize();
                if (flagPtr != nullptr) {
                    LocalTensor<float> f = diagT_.Get<float>();
                    f.SetValue(0, 1.0f);
                    PipeBarrier<PIPE_ALL>();
                    DataCopy(flagG_, f, 8);
                    PipeBarrier<PIPE_ALL>();
                }
            } else if (flagPtr != nullptr) {
                for (uint64_t spin = 0; spin < 200000000ull; ++spin) {
                    if (flagG_.GetValue(0) > 0.5f) break;
                    DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE,
                                             DcciDst::CACHELINE_OUT>(flagG_);
                }
            }
        }

        // ---- 3) full-N Add + RMSNorm on the reduced partial -------------------
        // norm_from_residual=1 is a diagnostic-only path: the residual input
        // slot carries an externally supplied partial, the residual add is
        // skipped, so the norm consumes exactly the tensor the caller passed.
        // Without it there is no way to inject a partial: with mm_enable=0 the
        // norm would read partialGm_, which is framework workspace that nothing
        // has written.
        // norm_from_x=1 binds the norm input straight to the x operand and
        // keeps the residual add, so the caller can feed an arbitrary
        // [M,N] partial in the x slot and compare against the reference with
        // the identical r/gamma/eps.  norm_from_residual=1 instead uses the
        // residual slot as the whole partial and skips the add.  Both are
        // diagnostic-only; with neither set the norm reads the AllReduce
        // result, or the matmul output when no collective is published.
        GM_ADDR srcGm = partialGm_;
        if (tiling_->normFromX != 0u) {
            srcGm = xGm_;
        } else if (tiling_->normFromResidual != 0u) {
            srcGm = residualGm_;
        } else if (hcclMode_ == 1u && arGm_ != nullptr) {
            srcGm = arGm_;
        }
        x_.SetGlobalBuffer((__gm__ bfloat16_t*)srcGm, static_cast<uint64_t>(m_) * n_);
        PipeBarrier<PIPE_ALL>();
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(x_);
        PipeBarrier<PIPE_ALL>();

        if (GetBlockIdx() == 0) {
            LocalTensor<float> d = diagT_.Get<float>();
            d.SetValue(0, static_cast<float>(rankId));
            d.SetValue(1, static_cast<float>(rankDim));
            d.SetValue(2, static_cast<float>(arRc));
            d.SetValue(3, hcclMode_ == 1u ? 1.0f : (hcclMode_ >= 1u ? 2.0f : 0.0f));
            d.SetValue(4, static_cast<float>(mmRc));
            PipeBarrier<PIPE_ALL>();
            DataCopy(diag_[0], d, 8);
            PipeBarrier<PIPE_ALL>();
        }

        LocalTensor<bfloat16_t> gl = bg_.Get<bfloat16_t>();
        LocalTensor<float> gf = bt_.Get<float>();
        DataCopy(gl, g_[0], n_);
        PipeBarrier<PIPE_ALL>();
        Cast(gf, gl, RoundMode::CAST_NONE, n_);
        PipeBarrier<PIPE_ALL>();

        const uint32_t blocks = GetBlockNum();
        const uint32_t perCoreRaw = (m_ + blocks - 1) / blocks;
        const uint32_t perCore = (perCoreRaw + 7u) / 8u * 8u;
        const uint32_t start = GetBlockIdx() * perCore;
        if (start >= m_) return;
        const uint32_t end = (start + perCore < m_) ? (start + perCore) : m_;

        LocalTensor<float> sums = bs_.Get<float>();
        LocalTensor<float> work = bwr_.Get<float>();
        uint32_t buffered = 0;
        uint32_t batchStart = start;

        for (uint32_t row = start; row < end; ++row) {
            const uint64_t base = static_cast<uint64_t>(row) * n_;
            LocalTensor<bfloat16_t> xl = bx_.Get<bfloat16_t>();
            LocalTensor<bfloat16_t> rl = br_.Get<bfloat16_t>();
            DataCopy(xl, x_[base], n_);
            DataCopy(rl, r_[base], n_);
            PipeBarrier<PIPE_ALL>();

            LocalTensor<float> z = bz_.Get<float>();
            LocalTensor<float> tmp = bw_.Get<float>();
            Cast(z, xl, RoundMode::CAST_NONE, n_);
            // The add is skipped only when the residual slot is itself the
            // partial, i.e. norm_from_residual is set AND it actually won the
            // source precedence.  norm_from_x keeps the add.
            if (!(tiling_->normFromResidual != 0u && tiling_->normFromX == 0u)) {
                Cast(tmp, rl, RoundMode::CAST_NONE, n_);
                Add(z, z, tmp, n_);
            }

            LocalTensor<bfloat16_t> rol = bo_.Get<bfloat16_t>();
            Cast(rol, z, RoundMode::CAST_RINT, n_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(ro_[base], rol, n_);
            PipeBarrier<PIPE_ALL>();

            Mul(tmp, z, z, n_);
            // Single-pass reduction over the whole row.  The previous 512-element
            // chunking split the summation tree arbitrarily on top of whatever
            // tree ReduceSum already uses, adding divergence that the native
            // reference does not have; the row is already resident in UB, so
            // there is no reason to chunk it.
            LocalTensor<float> part = bgst_.Get<float>();
            ReduceSum<float>(part, tmp, work, n_);
            const float acc = part.GetValue(0);
            const float rstd = 1.0f / sqrt(acc * invN_ + eps_);
            sums.SetValue(buffered, rstd);
            ++buffered;

            PipeBarrier<PIPE_ALL>();
            // Native AddRmsNorm sequence (add_rms_norm.h:233-244, same shape in
            // add_rms_norm_single_n.h and add_rms_norm_multi_n.h): normalise in
            // fp32, round to bf16, read back to fp32, multiply by gamma in
            // fp32, then round to bf16 once more.  The intermediate bf16
            // round-trip is part of the reference semantics; computing
            // z*rstd*gamma entirely in fp32 and rounding once is a different
            // evaluation and shifts elements that sit near a rounding
            // boundary.
            Muls(z, z, rstd, n_);
            PipeBarrier<PIPE_ALL>();
            LocalTensor<bfloat16_t> ol = bout_.Get<bfloat16_t>();
            Cast(ol, z, RoundMode::CAST_RINT, n_);   // fp32 -> bf16 (native :236)
            PipeBarrier<PIPE_ALL>();
            Cast(z, ol, RoundMode::CAST_NONE, n_);   // bf16 -> fp32 (native :238)
            PipeBarrier<PIPE_ALL>();
            Mul(z, z, gf, n_);                       // * gamma in fp32 (native :242)
            PipeBarrier<PIPE_ALL>();
            Cast(ol, z, RoundMode::CAST_RINT, n_);   // fp32 -> bf16 final (native :244)
            PipeBarrier<PIPE_ALL>();
            DataCopy(o_[base], ol, n_);
            PipeBarrier<PIPE_ALL>();

            if (buffered == 8 || row + 1 == end) {
                for (uint32_t k = buffered; k < 8; ++k) sums.SetValue(k, 0.0f);
                uint32_t cnt = 8;
                if (batchStart + cnt > mPad_) cnt = (mPad_ > batchStart) ? (mPad_ - batchStart) : 0;
                PipeBarrier<PIPE_ALL>();
                if (cnt > 0) DataCopy(rs_[batchStart], sums, cnt);
                PipeBarrier<PIPE_ALL>();
                batchStart += buffered;
                buffered = 0;
            }
        }
    }

private:
    TPipe pipe_;
    Matmul<AT, BT, CT, BiasT> mm_;
    TBuf<TPosition::VECCALC> bx_, br_, bg_, bo_, bout_, bz_, bt_, bw_, bs_, bwr_, bgst_, diagT_;
    GlobalTensor<bfloat16_t> x_, r_, g_, o_, ro_;
    GlobalTensor<float> rs_, diag_, flagG_;
    uint32_t m_{0}, n_{0}, k_{0};
    float eps_{1e-6f};
    float invN_{0.0f};
    uint32_t mPad_{0};
    ClefMatmulArAddRmsNormTilingData* tiling_{nullptr};
    uint32_t hcclMode_{0};
    bool normBuffersReady_{false};
    GM_ADDR xGm_{nullptr};
    GM_ADDR wGm_{nullptr};
    GM_ADDR wsGm_{nullptr};
    GM_ADDR partialGm_{nullptr};
    GM_ADDR arGm_{nullptr};
    GM_ADDR residualGm_{nullptr};
    __gm__ float* flagGm_{nullptr};
    uint8_t repeat_{1};
};

extern "C" __global__ __aicore__ void clef_matmul_ar_add_rms_norm_mc(
    GM_ADDR x, GM_ADDR w, GM_ADDR residual, GM_ADDR gamma,
    GM_ADDR out, GM_ADDR residualOut, GM_ADDR rstd, GM_ADDR diag,
    GM_ADDR workspace, GM_ADDR tilingGm) {
    REGISTER_TILING_DEFAULT(ClefMatmulArAddRmsNormTilingData);
    GET_TILING_DATA(tilingData, tilingGm);
    ClefMatmulArAddRmsNormKernel kernel;
    kernel.SetTiling(&tilingData);
    kernel.Init(x, w, residual, gamma, out, residualOut, rstd, diag, workspace, &tilingData);
    kernel.Process();
}
