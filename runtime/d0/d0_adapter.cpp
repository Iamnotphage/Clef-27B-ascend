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
#include <torch_npu/csrc/core/NPUStorageImpl.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <string>
#include <tuple>
#include <vector>
#include <cstdio>

#include "aclnn_clef_d0_trace.h"

namespace {

// Scope guard for the aclTensor handles: released on any exit path, including a
// throw from malloc or from OpCommand::Run. The workspace is NOT freed here; it
// is a per-process static cache reused across calls (see below).
struct TensorSet {
    aclTensor* h[9] = {nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr};
    ~TensorSet() {
        for (auto* t : h) if (t != nullptr) aclDestroyTensor(t);
    }
    void set(int i, aclTensor* t) { h[i] = t; }
};

struct DevicePrecondition {
    static void check(const at::Tensor& x, const at::Tensor& w, const at::Tensor& r,
                      const at::Tensor& g, const at::Tensor& trace) {
        TORCH_CHECK(x.device().type() == at::kPrivateUse1, "x must be on NPU");
        for (const auto* t : {&w, &r, &g}) {
            TORCH_CHECK(t->device() == x.device(),
                        "tensor on ", t->device(), " but x on ", x.device());
        }
        for (const auto* t : {&x, &w, &r, &g}) {
            TORCH_CHECK(t->scalar_type() == at::kBFloat16, "inputs must be bfloat16");
        }
        TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && r.dim() == 2 && g.dim() == 1,
                    "expected x/w/r 2-D and gamma 1-D");
        TORCH_CHECK(x.size(1) == w.size(0), "x K ", x.size(1), " != w K ", w.size(0));
        TORCH_CHECK(r.size(0) == x.size(0), "residual rows ", r.size(0), " != ", x.size(0));
        TORCH_CHECK(r.size(1) == w.size(1), "residual cols ", r.size(1), " != ", w.size(1));
        TORCH_CHECK(g.size(0) == w.size(1), "gamma len ", g.size(0), " != N ", w.size(1));
        TORCH_CHECK(trace.scalar_type() == at::kInt, "trace container must be int32");
        TORCH_CHECK(trace.dim() == 2 && trace.size(0) == 72 && trace.size(1) == 8,
                    "trace must be [72,8]");
        TORCH_CHECK(trace.numel() * trace.element_size() == 2304, "trace must be 2304 B");
        TORCH_CHECK(trace.device() == x.device(), "trace must live beside the inputs");
    }
};

constexpr int64_t kTraceSlots = 72;
constexpr int64_t kTraceFields = 8;

// Same descriptor read the E adapter uses: logical view dims and strides are
// kept while format and storage dims come from the tensor's real NPU storage.
// Nothing is forced to ND here.
aclTensor* mk(const at::Tensor& t, aclDataType dt, bool require_nz = false) {
    std::vector<int64_t> d(t.sizes().begin(), t.sizes().end());
    std::vector<int64_t> s(t.strides().begin(), t.strides().end());
    const auto* storage = static_cast<const torch_npu::NPUStorageImpl*>(
        t.storage().unsafeGetStorageImpl());
    const auto& desc = storage->npu_desc_;
    std::vector<int64_t> st(desc.storage_sizes_.begin(), desc.storage_sizes_.end());
    aclFormat fmt = static_cast<aclFormat>(desc.npu_format_);
    if (require_nz) {
        // W is NZ-only here: a base-format descriptor is refused before launch.
        TORCH_CHECK(fmt == ACL_FORMAT_FRACTAL_NZ,
                    "D0 requires an FRACTAL_NZ weight, got descriptor format=", (int)fmt);
        fprintf(stderr, "[d0] w descriptor format=%d storage=[", (int)fmt);
        for (size_t i = 0; i < st.size(); ++i)
            fprintf(stderr, "%lld%s", (long long)st[i], i + 1 < st.size() ? "," : "");
        fprintf(stderr, "] strides=[%lld,%lld] offset=%lld\n",
                (long long)s[0], (long long)s[1], (long long)t.storage_offset());
    }
    return aclCreateTensor(d.data(), static_cast<uint64_t>(d.size()), dt, s.data(),
                           t.storage_offset(), fmt, st.data(),
                           static_cast<uint64_t>(st.size()),
                           const_cast<void*>(t.storage().data()));
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
    // One allocation, never reused for anything else. The container dtype is
    // int32 because that is what this stack offers, but the aclTensor view below
    // is ACL_UINT32; the readback normalises through & 0xFFFFFFFF so values such
    // as magic 0xD0D0C0DE and seal 0xC0FFEE00 never appear negative.
    auto trace = at::empty({kTraceSlots, kTraceFields},
                           x.options().dtype(at::kInt).device(x.device()));

    // Preconditions run after the output tensors are allocated but before any
    // storage descriptor is read, so a mismatch cannot reach the cast.
    DevicePrecondition::check(x, w, residual, gamma, trace);

    TensorSet ts;                       // releases everything on any exit path
    ts.set(0, mk(x, ACL_BF16));
    ts.set(1, mk(w, ACL_BF16, /*require_nz=*/true));
    ts.set(2, mk(residual, ACL_BF16));
    ts.set(3, mk(gamma, ACL_BF16));
    ts.set(4, mk(out, ACL_BF16));
    ts.set(5, mk(res_out, ACL_BF16));
    ts.set(6, mk(rstd, ACL_FLOAT));
    ts.set(7, mk(diag, ACL_FLOAT));
    ts.set(8, mk(trace, ACL_UINT32));   // int32 storage seen as uint32
    aclTensor* tx = ts.h[0]; aclTensor* tw = ts.h[1]; aclTensor* tr = ts.h[2];
    aclTensor* tg = ts.h[3]; aclTensor* to = ts.h[4]; aclTensor* tro = ts.h[5];
    aclTensor* trs = ts.h[6]; aclTensor* td = ts.h[7]; aclTensor* ttr = ts.h[8];

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
        TORCH_CHECK(false, "d0 GetWorkspaceSize failed status=", (int)st,
                    " msg=", aclGetRecentErrMsg());   // handles freed by ts
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
    fprintf(stderr, "[d0] stream=%p trace_bytes=%lld\n", (void*)stream,
            (long long)(kTraceSlots * kTraceFields * 4));

    // Clear and launch are issued inside the same handler on the same explicit
    // stream, so the order clear -> launch cannot be reordered away.
    at_npu::native::OpCommand cmd;
    cmd.Name("ClefD0Trace");
    cmd.SetCustomHandler([ws, ws_size, ex, stream, trace]() -> int {
        const size_t nbytes = static_cast<size_t>(kTraceSlots * kTraceFields * 4);
        // aclrtMemsetAsync(void* dst, size_t destMax, int32_t value, size_t count, stream)
        aclError clr = aclrtMemsetAsync(trace.data_ptr(), nbytes, 0, nbytes, stream);
        if (clr != ACL_SUCCESS) return static_cast<int>(clr);
        return static_cast<int>(aclnnClefD0Trace(ws, ws_size, ex, stream));
    });
    cmd.Run();

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
