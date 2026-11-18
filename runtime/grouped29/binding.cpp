#include "first_trace.h"
#include <ATen/ATen.h>
#include <torch/library.h>
#include <acl/acl.h>
#include <aclnnop/aclnn_grouped_mat_mul_all_reduce.h>
#include <torch_npu/csrc/core/NPUStorageImpl.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace {
constexpr const char* OPAPI = "/usr/local/Ascend/cann-9.2.1/aarch64-linux/lib64/libopapi.so";
constexpr const char* META = "/usr/local/Ascend/cann-9.2.1/aarch64-linux/lib64/libnnopbase.so";

std::string canonical(const char* path) {
    std::unique_ptr<char, decltype(&std::free)> p(realpath(path, nullptr), std::free);
    TORCH_CHECK(p, "grouped29 realpath failed: ", path);
    return p.get();
}
template <class F> F symbol(void* lib, const char* name, const char* expected) {
    dlerror();
    void* p = dlsym(lib, name);
    const char* error = dlerror();
    TORCH_CHECK(p && !error, "grouped29 missing symbol: ", name, " ", error ? error : "");
    Dl_info info{};
    TORCH_CHECK(dladdr(p, &info) && info.dli_fname && canonical(info.dli_fname) == canonical(expected),
                "grouped29 unexpected symbol provider: ", name);
    return reinterpret_cast<F>(p);
}
struct Api {
    void* lib;
    void* meta;
    decltype(&aclnnGroupedMatMulAllReduceGetWorkspaceSize) prepare;
    decltype(&aclnnGroupedMatMulAllReduce) launch;
    decltype(&aclCreateTensor) create_tensor;
    decltype(&aclCreateTensorList) create_list;
    decltype(&aclDestroyTensor) destroy_tensor;
    decltype(&aclDestroyTensorList) destroy_list;
    Api() {
        lib = dlopen(OPAPI, RTLD_NOW | RTLD_LOCAL);
        TORCH_CHECK(lib, "grouped29 opapi dlopen: ", dlerror());
        meta = dlopen(META, RTLD_NOW | RTLD_LOCAL);
        TORCH_CHECK(meta, "grouped29 metadata dlopen: ", dlerror());
        prepare = symbol<decltype(prepare)>(lib, "aclnnGroupedMatMulAllReduceGetWorkspaceSize", OPAPI);
        launch = symbol<decltype(launch)>(lib, "aclnnGroupedMatMulAllReduce", OPAPI);
        create_tensor = symbol<decltype(create_tensor)>(meta, "aclCreateTensor", META);
        create_list = symbol<decltype(create_list)>(meta, "aclCreateTensorList", META);
        destroy_tensor = symbol<decltype(destroy_tensor)>(meta, "aclDestroyTensor", META);
        destroy_list = symbol<decltype(destroy_list)>(meta, "aclDestroyTensorList", META);
        // Keep providers mapped until process exit, including queued launches.
    }
};
Api& api() { static Api value; return value; }

struct Lists {
    Api& a;
    std::vector<aclTensorList*> lists;
    explicit Lists(Api& value) : a(value) {}
    int close() noexcept {
        int first_error = 0;
        for (auto* list : lists) {
            int rc = a.destroy_list(list);  // Owns/destroys its member descriptors.
            if (rc && !first_error) first_error = rc;
        }
        lists.clear();
        return first_error;
    }
    ~Lists() {
        const int rc = close();
        if (rc) std::fprintf(stderr, "grouped29 exceptional cleanup rc=%d\n", rc);
    }
    aclTensorList* add(const at::Tensor& t, aclDataType dtype) {
        const auto* storage = static_cast<const torch_npu::NPUStorageImpl*>(t.storage().unsafeGetStorageImpl());
        const auto& desc = storage->npu_desc_;
        TORCH_CHECK(static_cast<aclFormat>(desc.npu_format_) == ACL_FORMAT_ND,
                    "grouped29 requires actual ND storage, got ", static_cast<int>(desc.npu_format_));
        std::vector<int64_t> sizes(t.sizes().begin(), t.sizes().end());
        std::vector<int64_t> strides(t.strides().begin(), t.strides().end());
        std::vector<int64_t> storage_sizes(desc.storage_sizes_.begin(), desc.storage_sizes_.end());
        auto* tensor = a.create_tensor(sizes.data(), sizes.size(), dtype, strides.data(), t.storage_offset(),
            ACL_FORMAT_ND, storage_sizes.data(), storage_sizes.size(), const_cast<void*>(t.storage().data()));
        TORCH_CHECK(tensor, "grouped29 aclCreateTensor returned null");
        const aclTensor* members[] = {tensor};
        auto* list = a.create_list(members, 1);
        if (!list) {
            const int rc = a.destroy_tensor(tensor);
            TORCH_CHECK(false, "grouped29 aclCreateTensorList returned null, cleanup rc=", rc);
        }
        lists.push_back(list); // Exactly one owner, no standalone tensor destruction.
        return list;
    }
};
}

std::tuple<at::Tensor, int64_t> grouped29(
    const at::Tensor& x, const at::Tensor& weight_kn, const at::Tensor& bias,
    const std::string& group, bool poison) {
    std::shared_ptr<grouped29_trace::FirstCallTrace> trace;
    if (poison) {
        trace = std::make_shared<grouped29_trace::FirstCallTrace>();
        trace->mark("binding_enter");
    }
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "grouped29 requires NPU input");
    c10_npu::NPUGuard guard(x.device());
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 16384 && x.size(1) == 8704,
                "grouped29 x must be [M,8704]");
    TORCH_CHECK(weight_kn.dim() == 2 && weight_kn.size(0) == 8704 && weight_kn.size(1) == 5120,
                "grouped29 weight must be contiguous ND [8704,5120]");
    for (const auto* t : {&x, &weight_kn}) {
        TORCH_CHECK(t->device() == x.device() && t->scalar_type() == at::kBFloat16 &&
                    t->is_contiguous() && t->storage_offset() == 0, "grouped29 BF16/input layout mismatch");
    }
    TORCH_CHECK(bias.device() == x.device() && bias.scalar_type() == at::kFloat &&
                bias.dim() == 1 && bias.size(0) == 5120 && bias.is_contiguous() && bias.storage_offset() == 0,
                "grouped29 requires explicit contiguous FP32 bias [5120]");
    TORCH_CHECK(!group.empty() && group.find('\0') == std::string::npos, "grouped29 invalid HCCL group");
    Api& a = api();
    auto y = at::empty({x.size(0), 5120}, x.options());
    if (poison) y.fill_(std::numeric_limits<float>::quiet_NaN());
    if (trace) trace->mark("descriptors_enter");
    auto lists = std::make_shared<Lists>(a);
    auto* tx = lists->add(x, ACL_BF16);
    auto* tw = lists->add(weight_kn, ACL_BF16);
    auto* tb = lists->add(bias, ACL_FLOAT);
    auto* ty = lists->add(y, ACL_BF16);
    if (trace) trace->mark("descriptors_ready");
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    if (trace) trace->mark("workspace_enter");
    const auto rc = a.prepare(tx, tw, tb, nullptr, 0, group.c_str(), "sum", 0, 1, ty, &bytes, &executor);
    if (trace) trace->workspace(rc, bytes, executor != nullptr);
    if (rc != 0 || !executor) {
        const std::string error = aclGetRecentErrMsg() ? aclGetRecentErrMsg() : "";
        const int cleanup = lists->close();
        if (trace) trace->rc("prepare_cleanup_return", cleanup);
        TORCH_CHECK(false, "grouped29 prepare rc=", rc, " executor=", executor,
                    " cleanup rc=", cleanup, " ", error);
    }
    TORCH_CHECK(bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), "workspace overflow");
    auto workspace = at::empty({static_cast<int64_t>(bytes)}, x.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream().stream();
    auto launch = a.launch;
    at_npu::native::OpCommand cmd;
    cmd.Name("ClefGroupedMmar29");
    cmd.SetCustomHandler([lists, workspace, x, weight_kn, bias, y, executor, bytes, stream, launch, trace]() -> int {
        if (trace) trace->mark("handler_enter");
        if (trace) trace->mark("launch_enter");
        const int launch_rc = launch(bytes ? workspace.data_ptr() : nullptr, bytes, executor, stream);
        if (trace) trace->rc("launch_return", launch_rc);
        const std::string error = launch_rc && aclGetRecentErrMsg() ? aclGetRecentErrMsg() : "";
        const int cleanup_rc = lists->close();
        if (trace) trace->rc("launch_cleanup_return", cleanup_rc);
        TORCH_CHECK(launch_rc == 0 && cleanup_rc == 0,
                    "grouped29 launch rc=", launch_rc, " cleanup rc=", cleanup_rc, " ", error);
        if (trace) trace->rc("handler_return", 0);
        return 0;
    });
    if (trace) trace->mark("dispatch_enter");
    cmd.Run();
    if (trace) trace->mark("dispatch_return");
    return {y, static_cast<int64_t>(bytes)};
}

std::string grouped29_provider() {
    (void)api(); // All dlsym providers are checked using dladdr above.
    return std::string("prepare/launch=") + canonical(OPAPI) + ";metadata=" + canonical(META);
}
TORCH_LIBRARY(clef_grouped29, m) {
    m.def("run(Tensor x, Tensor weight_kn, Tensor bias, str group, bool poison=True) -> (Tensor, int)");
    m.def("provider() -> str", &grouped29_provider);
}
TORCH_LIBRARY_IMPL(clef_grouped29, PrivateUse1, m) { m.impl("run", &grouped29); }
