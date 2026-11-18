// Torch adapter for ClefArAddRmsNorm.
//
// It calls the generated aclnn two-phase API directly from the torch process:
// the generated inner GetWorkspaceSize is what emits NnopbaseSetMc2 /
// NnopbaseSetHcomGroup (the MC2 marking), so no OpCommand handler is needed
// here. The group string must be the real hcom name of the live process
// group, because the MC2 context injected at launch is resolved from it.
#include <torch/extension.h>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <string>
#include <tuple>
#include <vector>
#include <cstdio>

#include "aclnn_clef_ar_add_rms_norm.h"

namespace {

aclTensor* make_tensor(const at::Tensor& t, aclDataType dtype) {
    std::vector<int64_t> dims(t.sizes().begin(), t.sizes().end());
    std::vector<int64_t> strides(t.strides().begin(), t.strides().end());
    return aclCreateTensor(dims.data(), static_cast<uint64_t>(dims.size()), dtype,
                           strides.data(), 0, ACL_FORMAT_ND, dims.data(),
                           static_cast<uint64_t>(dims.size()),
                           const_cast<void*>(t.data_ptr()));
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ar_add_rms_norm(
    const at::Tensor& x, const at::Tensor& residual, const at::Tensor& gamma,
    double eps, int64_t repeat, std::string group) {
    auto out = at::empty_like(x);
    auto res_out = at::empty_like(x);
    auto rstd = at::empty({x.size(0)}, x.options().dtype(at::kFloat));
    auto diag = at::empty({8}, x.options().dtype(at::kFloat));

    aclTensor* tx = make_tensor(x, ACL_BF16);
    aclTensor* tr = make_tensor(residual, ACL_BF16);
    aclTensor* tg = make_tensor(gamma, ACL_BF16);
    aclTensor* to = make_tensor(out, ACL_BF16);
    aclTensor* tro = make_tensor(res_out, ACL_BF16);
    aclTensor* trs = make_tensor(rstd, ACL_FLOAT);
    aclTensor* tdiag = make_tensor(diag, ACL_FLOAT);

    uint64_t workspace = 0;
    aclOpExecutor* executor = nullptr;
    char* group_ptr = const_cast<char*>(group.c_str());
    fprintf(stderr, "[adapter] before GetWorkspaceSize group=%s\n", group.c_str());
    aclnnStatus st = aclnnClefArAddRmsNormGetWorkspaceSize(
        tx, tr, tg, eps, repeat, group_ptr, to, tro, trs, tdiag, &workspace, &executor);
    fprintf(stderr, "[adapter] after GetWorkspaceSize st=%d ws=%llu exec=%p\n", (int)st, (unsigned long long)workspace, (void*)executor);
    TORCH_CHECK(st == 0 && executor != nullptr,
                "GetWorkspaceSize failed status=", static_cast<int>(st),
                " msg=", aclGetRecentErrMsg());
    void* ws = nullptr;
    if (workspace > 0) {
        TORCH_CHECK(aclrtMalloc(&ws, workspace, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
                    "workspace alloc failed");
    }
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    // Run through OpCommand so the Nnopbase launch path applies the MC2
    // resources that the generated GetWorkspaceSize marked.
    at_npu::native::OpCommand cmd;
    cmd.Name("ClefArAddRmsNorm");
    cmd.SetCustomHandler([ws, workspace, executor, stream]() -> int {
        return static_cast<int>(aclnnClefArAddRmsNorm(ws, workspace, executor, stream));
    });
    fprintf(stderr, "[adapter] before OpCommand.Run\n");
    cmd.Run();
    fprintf(stderr, "[adapter] after OpCommand.Run\n");
    aclDestroyTensor(tx);
    aclDestroyTensor(tr);
    aclDestroyTensor(tg);
    aclDestroyTensor(to);
    aclDestroyTensor(tro);
    aclDestroyTensor(trs);
    aclDestroyTensor(tdiag);
    return std::make_tuple(out, res_out, rstd, diag);
}

TORCH_LIBRARY(clef_ar, m) {
    m.def("ar_add_rms_norm(Tensor x, Tensor residual, Tensor gamma, float eps, "
          "int repeat, str group) -> (Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(clef_ar, PrivateUse1, m) {
    m.impl("ar_add_rms_norm", &ar_add_rms_norm);
}
