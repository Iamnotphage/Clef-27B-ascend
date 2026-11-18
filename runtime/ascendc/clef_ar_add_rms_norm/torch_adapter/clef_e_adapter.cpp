// Torch adapter for 22's ClefMatmulArAddRmsNorm (E first version).
#include <torch/extension.h>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/NPUStorageImpl.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <string>
#include <tuple>
#include <vector>
#include "aclnn_clef_matmul_ar_add_rms_norm.h"

namespace {
aclTensor* mk(const at::Tensor& t, aclDataType dt, bool nz = false) {
    // Faithful port of the stack's op_api_common.h ConvertType(at::Tensor):
    // logical view dims and strides are kept, while format and storage dims
    // come from the tensor's real NPU storage descriptor.
    std::vector<int64_t> d(t.sizes().begin(), t.sizes().end());
    std::vector<int64_t> s(t.strides().begin(), t.strides().end());
    // Header-only bridge (see the stack's NPUBridge.cpp:21-23): the
    // installed torch_npu does not export GetNpuStorageImpl as a symbol.
    const auto* storage = static_cast<const torch_npu::NPUStorageImpl*>(
        t.storage().unsafeGetStorageImpl());
    const auto& desc = storage->npu_desc_;
    std::vector<int64_t> st(desc.storage_sizes_.begin(), desc.storage_sizes_.end());
    aclFormat fmt = nz ? static_cast<aclFormat>(desc.npu_format_) : ACL_FORMAT_ND;
    if (nz) {
        fprintf(stderr, "[e-adapter] NZ descriptor format=%lld storage=[%lld,%lld,%lld,%lld] "
                        "strides=[%lld,%lld] offset=%lld\n",
                (long long)desc.npu_format_,
                (long long)(st.size() > 0 ? st[0] : -1), (long long)(st.size() > 1 ? st[1] : -1),
                (long long)(st.size() > 2 ? st[2] : -1), (long long)(st.size() > 3 ? st[3] : -1),
                (long long)s[0], (long long)s[1], (long long)t.storage_offset());
    }
    return aclCreateTensor(d.data(), (uint64_t)d.size(), dt, s.data(), t.storage_offset(), fmt,
                           st.data(), (uint64_t)st.size(),
                           const_cast<void*>(t.storage().data()));
}
}
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mm_ar_norm(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& residual,
    const at::Tensor& gamma, double eps, int64_t repeat, std::string group, int64_t use_hccl, int64_t block_dim, int64_t mm_enable, int64_t mm_dim, int64_t mm_mode, int64_t mm_bufspace, int64_t norm_ub_late, int64_t mm_btrans, int64_t mm_step, int64_t norm_from_residual, int64_t norm_from_x, int64_t w_nz) {
    auto out = at::empty({x.size(0), w.size(1)}, x.options());
    auto res_out = at::empty_like(out);
    auto rstd = at::empty({x.size(0)}, x.options().dtype(at::kFloat));
    auto diag = at::empty({8}, x.options().dtype(at::kFloat));
    aclTensor *tx = mk(x, ACL_BF16), *tw = mk(w, ACL_BF16, w_nz != 0), *tr = mk(residual, ACL_BF16),
             *tg = mk(gamma, ACL_BF16), *to = mk(out, ACL_BF16), *tro = mk(res_out, ACL_BF16),
             *trs = mk(rstd, ACL_FLOAT), *td = mk(diag, ACL_FLOAT);
    uint64_t ws_size = 0;
    aclOpExecutor* ex = nullptr;
    char* g = const_cast<char*>(group.c_str());
    aclnnStatus st = aclnnClefMatmulArAddRmsNormGetWorkspaceSize(
        tx, tw, tr, tg, eps, repeat, g, use_hccl, block_dim, mm_enable, mm_dim, mm_mode, mm_bufspace, norm_ub_late, mm_btrans, mm_step, norm_from_residual, norm_from_x, to, tro, trs, td, &ws_size, &ex);
    TORCH_CHECK(st == 0 && ex != nullptr, "E GetWorkspaceSize failed ", (int)st, " ", aclGetRecentErrMsg());
    fprintf(stderr, "[e-adapter] after GetWorkspaceSize ws=%llu\n", (unsigned long long)ws_size);
    void* ws = nullptr;
    static void* cached = nullptr; static uint64_t cached_sz = 0;
    if (ws_size > 0) {
        if (!cached || cached_sz < ws_size) { if (cached) aclrtFree(cached);
            TORCH_CHECK(aclrtMalloc(&cached, ws_size, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "ws alloc"); cached_sz = ws_size; }
        ws = cached;
    }
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("ClefMatmulArAddRmsNorm");
    cmd.SetCustomHandler([ws, ws_size, ex, stream]() -> int {
        return (int)aclnnClefMatmulArAddRmsNorm(ws, ws_size, ex, stream); });
    cmd.Run();
    aclDestroyTensor(tx); aclDestroyTensor(tw); aclDestroyTensor(tr); aclDestroyTensor(tg);
    aclDestroyTensor(to); aclDestroyTensor(tro); aclDestroyTensor(trs); aclDestroyTensor(td);
    return {out, res_out, rstd, diag};
}
TORCH_LIBRARY(clef_e, m) {
    m.def("mm_ar_norm(Tensor x, Tensor w, Tensor residual, Tensor gamma, float eps, int repeat, str group, int use_hccl, int block_dim, int mm_enable, int mm_dim, int mm_mode, int mm_bufspace, int norm_ub_late, int mm_btrans, int mm_step, int nfr, int nfx, int w_nz) -> (Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(clef_e, PrivateUse1, m) { m.impl("mm_ar_norm", &mm_ar_norm); }
