// Standalone torch adapter for the D0 diagnostic op (ClefD0Trace).
//
// Independent of the E adapter: own file, own TORCH_LIBRARY, own symbols.
// ABI taken from the generated header read on the build host:
//   4 inputs  : x, w, residual, gamma
//   14 attrs  : eps, repeat, group, useHccl, blockDim, mmEnable, mmDim,
//               mmMode, mmBufspace, normUbLate, mmBtrans, mmStage,
//               normFromResidual, normFromX
//   5 outputs : out, residual_out, rstd, diag, trace
//   trace     : uint32, ND, [72, 8] = 2304 B, allocated once by this adapter
//
// Ordering contract for the trace buffer (host side, not the kernel):
//   clear on the stream -> launch -> synchronize -> read back.
// The kernel only ever writes its own slot, so the clear must be ordered
// before it on the same stream; both happen through that one stream here.
#include <torch/extension.h>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <string>
#include <tuple>
#include <vector>
#include <cstdio>

#include "aclnn_clef_d0_trace.h"

namespace {

constexpr int64_t kTraceSlots = 72;
constexpr int64_t kTraceFields = 8;

aclTensor* mk(const at::Tensor& t, aclDataType dt) {
    std::vector<int64_t> d(t.sizes().begin(), t.sizes().end());
    std::vector<int64_t> s(t.strides().begin(), t.strides().end());
    return aclCreateTensor(d.data(), static_cast<uint64_t>(d.size()), dt, s.data(), 0,
                           ACL_FORMAT_ND, d.data(), static_cast<uint64_t>(d.size()),
                           const_cast<void*>(t.data_ptr()));
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> d0_trace(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& residual,
    const at::Tensor& gamma, double eps, int64_t repeat, std::string group,
    int64_t use_hccl, int64_t block_dim, int64_t mm_enable, int64_t mm_dim,
    int64_t mm_mode, int64_t mm_bufspace, int64_t norm_ub_late, int64_t mm_btrans,
    int64_t mm_stage, int64_t norm_from_residual, int64_t norm_from_x) {
    auto out = at::empty({x.size(0), w.size(1)}, x.options());
    auto res_out = at::empty_like(out);
    auto rstd = at::empty({x.size(0)}, x.options().dtype(at::kFloat));
    auto diag = at::empty({8}, x.options().dtype(at::kFloat));
    // One allocation, never reused for anything else.
    auto trace = at::zeros({kTraceSlots, kTraceFields},
                           x.options().dtype(at::kInt).device(x.device()));

    aclTensor* tx = mk(x, ACL_BF16);
    aclTensor* tw = mk(w, ACL_BF16);
    aclTensor* tr = mk(residual, ACL_BF16);
    aclTensor* tg = mk(gamma, ACL_BF16);
    aclTensor* to = mk(out, ACL_BF16);
    aclTensor* tro = mk(res_out, ACL_BF16);
    aclTensor* trs = mk(rstd, ACL_FLOAT);
    aclTensor* td = mk(diag, ACL_FLOAT);
    aclTensor* ttr = mk(trace, ACL_UINT32);

    uint64_t ws_size = 0;
    aclOpExecutor* ex = nullptr;
    char* g = const_cast<char*>(group.c_str());
    fprintf(stderr, "[d0] before GetWorkspaceSize group=%s mm_dim=%lld use_hccl=%lld\n",
            group.c_str(), (long long)mm_dim, (long long)use_hccl);
    aclnnStatus st = aclnnClefD0TraceGetWorkspaceSize(
        tx, tw, tr, tg, eps, repeat, g, use_hccl, block_dim, mm_enable, mm_dim,
        mm_mode, mm_bufspace, norm_ub_late, mm_btrans, mm_stage,
        norm_from_residual, norm_from_x, to, tro, trs, td, ttr, &ws_size, &ex);
    fprintf(stderr, "[d0] after GetWorkspaceSize st=%d ws=%llu\n", (int)st,
            (unsigned long long)ws_size);
    if (st != 0 || ex == nullptr) {
        aclDestroyTensor(tx); aclDestroyTensor(tw); aclDestroyTensor(tr);
        aclDestroyTensor(tg); aclDestroyTensor(to); aclDestroyTensor(tro);
        aclDestroyTensor(trs); aclDestroyTensor(td); aclDestroyTensor(ttr);
        TORCH_CHECK(false, "d0 GetWorkspaceSize failed status=", (int)st,
                    " msg=", aclGetRecentErrMsg());
    }

    static void* cached_ws = nullptr;
    static uint64_t cached_size = 0;
    void* ws = nullptr;
    if (ws_size > 0) {
        if (cached_ws == nullptr || cached_size < ws_size) {
            if (cached_ws != nullptr) aclrtFree(cached_ws);
            TORCH_CHECK(aclrtMalloc(&cached_ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
                        "d0 workspace alloc failed");
            cached_size = ws_size;
        }
        ws = cached_ws;
    }
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    // The clear above ran when the tensor was zero-filled; both it and the
    // launch sit on this one stream, so the order is clear -> launch.
    at_npu::native::OpCommand cmd;
    cmd.Name("ClefD0Trace");
    cmd.SetCustomHandler([ws, ws_size, ex, stream]() -> int {
        return static_cast<int>(aclnnClefD0Trace(ws, ws_size, ex, stream));
    });
    cmd.Run();

    aclDestroyTensor(tx); aclDestroyTensor(tw); aclDestroyTensor(tr);
    aclDestroyTensor(tg); aclDestroyTensor(to); aclDestroyTensor(tro);
    aclDestroyTensor(trs); aclDestroyTensor(td); aclDestroyTensor(ttr);

    // Caller synchronizes the stream, then reads trace back as [72, 8] uint32.
    return std::make_tuple(out, res_out, rstd, diag, trace);
}

TORCH_LIBRARY(clef_d0, m) {
    m.def("d0_trace(Tensor x, Tensor w, Tensor residual, Tensor gamma, float eps, "
          "int repeat, str group, int use_hccl, int block_dim, int mm_enable, "
          "int mm_dim, int mm_mode, int mm_bufspace, int norm_ub_late, int mm_btrans, "
          "int mm_stage, int norm_from_residual, int norm_from_x) "
          "-> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(clef_d0, PrivateUse1, m) {
    m.impl("d0_trace", &d0_trace);
}
