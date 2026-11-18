// Torch adapter for 22's ClefMatmulArAddRmsNorm (E first version).
#include <torch/extension.h>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <string>
#include <tuple>
#include <vector>
#include "aclnn_clef_matmul_ar_add_rms_norm.h"

namespace {
aclTensor* mk(const at::Tensor& t, aclDataType dt) {
    std::vector<int64_t> d(t.sizes().begin(), t.sizes().end());
    std::vector<int64_t> s(t.strides().begin(), t.strides().end());
    return aclCreateTensor(d.data(), (uint64_t)d.size(), dt, s.data(), 0, ACL_FORMAT_ND,
                           d.data(), (uint64_t)d.size(), const_cast<void*>(t.data_ptr()));
}
}
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mm_ar_norm(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& residual,
    const at::Tensor& gamma, double eps, int64_t repeat, std::string group, int64_t use_hccl, int64_t block_dim, int64_t mm_enable, int64_t mm_dim) {
    auto out = at::empty({x.size(0), w.size(1)}, x.options());
    auto res_out = at::empty_like(out);
    auto rstd = at::empty({x.size(0)}, x.options().dtype(at::kFloat));
    auto diag = at::empty({8}, x.options().dtype(at::kFloat));
    aclTensor *tx = mk(x, ACL_BF16), *tw = mk(w, ACL_BF16), *tr = mk(residual, ACL_BF16),
             *tg = mk(gamma, ACL_BF16), *to = mk(out, ACL_BF16), *tro = mk(res_out, ACL_BF16),
             *trs = mk(rstd, ACL_FLOAT), *td = mk(diag, ACL_FLOAT);
    uint64_t ws_size = 0;
    aclOpExecutor* ex = nullptr;
    char* g = const_cast<char*>(group.c_str());
    aclnnStatus st = aclnnClefMatmulArAddRmsNormGetWorkspaceSize(
        tx, tw, tr, tg, eps, repeat, g, use_hccl, block_dim, mm_enable, mm_dim, to, tro, trs, td, &ws_size, &ex);
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
    m.def("mm_ar_norm(Tensor x, Tensor w, Tensor residual, Tensor gamma, float eps, int repeat, str group, int use_hccl, int block_dim, int mm_enable, int mm_dim) -> (Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(clef_e, PrivateUse1, m) { m.impl("mm_ar_norm", &mm_ar_norm); }
