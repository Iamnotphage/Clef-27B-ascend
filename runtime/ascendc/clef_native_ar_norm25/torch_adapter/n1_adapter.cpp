#include <ATen/ATen.h>
#include <torch/library.h>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/NPUStorageImpl.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "aclnn_clef_native_ar_norm25.h"
#include <dlfcn.h>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace {
struct Api {
    std::string path;
    void* library;
    decltype(&aclnnClefNativeArNorm25GetWorkspaceSize) prepare;
    decltype(&aclnnClefNativeArNorm25) launch;
    explicit Api(const std::string& p) : path(p) {
        TORCH_CHECK(!p.empty() && p[0] == '/', "opapi path must be absolute");
        library = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
        TORCH_CHECK(library, "N1 opapi dlopen failed: ", dlerror());
        prepare = reinterpret_cast<decltype(prepare)>(dlsym(library, "aclnnClefNativeArNorm25GetWorkspaceSize"));
        launch = reinterpret_cast<decltype(launch)>(dlsym(library, "aclnnClefNativeArNorm25"));
        TORCH_CHECK(prepare && launch, "N1 generated symbols missing");
        // Retain library for outstanding asynchronous work until process exit.
    }
};
struct TensorSet {
    std::vector<aclTensor*> handles;
    ~TensorSet() { for (auto* h : handles) aclDestroyTensor(h); }
    aclTensor* add(const at::Tensor& t, aclDataType dt) {
        const auto* st = static_cast<const torch_npu::NPUStorageImpl*>(t.storage().unsafeGetStorageImpl());
        const auto& desc = st->npu_desc_;
        TORCH_CHECK(static_cast<aclFormat>(desc.npu_format_) == ACL_FORMAT_ND,
                    "N1 requires actual ND storage, format=", static_cast<int>(desc.npu_format_));
        std::vector<int64_t> dims(t.sizes().begin(), t.sizes().end());
        std::vector<int64_t> strides(t.strides().begin(), t.strides().end());
        std::vector<int64_t> storage(desc.storage_sizes_.begin(), desc.storage_sizes_.end());
        aclTensor* h = aclCreateTensor(dims.data(), dims.size(), dt, strides.data(),
            t.storage_offset(), ACL_FORMAT_ND, storage.data(), storage.size(),
            const_cast<void*>(t.storage().data()));
        TORCH_CHECK(h, "aclCreateTensor failed");
        handles.push_back(h);
        return h;
    }
};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> n1(
    const at::Tensor& partial, const at::Tensor& residual, const at::Tensor& gamma,
    double eps, const std::string& group, int64_t use_hccl, int64_t cores,
    const std::string& api_path, bool poison) {
    TORCH_CHECK(partial.device().type() == at::kPrivateUse1, "N1 expects NPU input");
    for (const auto* t : {&partial, &residual, &gamma}) {
        TORCH_CHECK(t->device() == partial.device() && t->scalar_type() == at::kBFloat16 &&
                    t->is_contiguous(), "N1 inputs must be contiguous BF16 on the same device");
    }
    TORCH_CHECK(partial.dim() == 2 && partial.size(0) >= 1 && partial.size(0) <= 16384 &&
                partial.size(1) == 5120 && residual.sizes() == partial.sizes() &&
                gamma.dim() == 1 && gamma.size(0) == 5120, "N1 shape mismatch");
    TORCH_CHECK(std::isfinite(eps) && eps > 0 && (use_hccl == 0 || use_hccl == 1) &&
                cores >= 1 && cores <= 48 && !group.empty(), "N1 attrs invalid");
    static Api api(api_path);
    TORCH_CHECK(api.path == api_path, "N1 opapi identity cannot change in a running process");
    auto y = at::empty(partial.sizes(), partial.options());
    auto res = at::empty(partial.sizes(), partial.options());
    auto rs = at::empty({partial.size(0)}, partial.options().dtype(at::kFloat));
    auto reduced = at::empty(partial.sizes(), partial.options());
    auto diag = at::zeros({48, 32}, partial.options().dtype(at::kInt));
    if (poison) {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        y.fill_(nan); res.fill_(nan); rs.fill_(nan); reduced.fill_(nan);
    }
    auto ts = std::make_shared<TensorSet>();
    auto* tp = ts->add(partial, ACL_BF16);
    auto* tr = ts->add(residual, ACL_BF16);
    auto* tg = ts->add(gamma, ACL_BF16);
    auto* ty = ts->add(y, ACL_BF16);
    auto* tres = ts->add(res, ACL_BF16);
    auto* trs = ts->add(rs, ACL_FLOAT);
    auto* tred = ts->add(reduced, ACL_BF16);
    auto* td = ts->add(diag, ACL_UINT32);
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    auto status = api.prepare(tp, tr, tg, eps, const_cast<char*>(group.c_str()), use_hccl,
                             cores, ty, tres, trs, tred, td, &bytes, &executor);
    TORCH_CHECK(status == 0 && executor, "N1 prepare status=", status, " ", aclGetRecentErrMsg());
    TORCH_CHECK(bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), "workspace overflow");
    // Torch allocator owns per-call workspace lifetime on the current stream.
    auto ws = at::empty({static_cast<int64_t>(bytes)}, partial.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream().stream();
    auto launch = api.launch;
    at_npu::native::OpCommand cmd;
    cmd.Name("ClefNativeArNorm25");
    cmd.SetCustomHandler([ts, ws, executor, bytes, stream, launch]() -> int {
        return static_cast<int>(launch(bytes ? ws.data_ptr() : nullptr, bytes, executor, stream));
    });
    cmd.Run();
    return {y, res, rs, reduced, diag};
}

TORCH_LIBRARY(clef_n1_25, m) {
    m.def("run(Tensor partial, Tensor residual, Tensor gamma, float eps, str group, "
          "int use_hccl, int cores, str api_path, bool poison=True) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(clef_n1_25, PrivateUse1, m) { m.impl("run", &n1); }
