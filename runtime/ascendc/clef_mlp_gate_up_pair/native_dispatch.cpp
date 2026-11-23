#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <aclnn/acl_meta.h>
#include <dlfcn.h>

#include <array>
#include <cstdlib>
#include <memory>

namespace {
constexpr int64_t Hidden = 5120;
constexpr int64_t Intermediate = 8704;
using WorkspaceGetter = int (*)(const aclTensor*, const aclTensor*, int64_t, const aclTensor*,
                                uint64_t*, aclOpExecutor**);
using Launcher = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);

void* OpenOpApi()
{
    const char* path = std::getenv("CLEF_GATE_UP_OPAPI");
    TORCH_CHECK(path && path[0] == '/', "CLEF_GATE_UP_OPAPI must be an absolute private library path");
    void* handle = dlopen(path, RTLD_LOCAL | RTLD_NOW);
    TORCH_CHECK(handle, "could not load private gate/up op_api");
    // The function pointers remain valid for the lifetime of this process.
    return handle;
}

template<typename Function>
Function LoadSymbol(void* handle, const char* name)
{
    auto symbol = dlsym(handle, name);
    TORCH_CHECK(symbol, "missing private ACLNN symbol ", name);
    return reinterpret_cast<Function>(symbol);
}

struct Api {
    void* handle = OpenOpApi();
    decltype(&aclCreateTensor) create = LoadSymbol<decltype(create)>(handle, "aclCreateTensor");
    decltype(&aclDestroyTensor) destroy = LoadSymbol<decltype(destroy)>(handle, "aclDestroyTensor");
    WorkspaceGetter getWorkspace =
        LoadSymbol<WorkspaceGetter>(handle, "aclnnClefMlpGateUpPairGetWorkspaceSize");
    Launcher launch = LoadSymbol<Launcher>(handle, "aclnnClefMlpGateUpPair");
};

using Descriptor = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;

Descriptor MakeDescriptor(const Api& api, const at::Tensor& tensor, aclFormat format,
                          at::IntArrayRef storageShape)
{
    Descriptor result(api.create(tensor.sizes().data(), tensor.dim(), ACL_BF16,
                                 tensor.strides().data(), 0, format, storageShape.data(),
                                 storageShape.size(), tensor.data_ptr()), api.destroy);
    TORCH_CHECK(result, "aclCreateTensor failed");
    return result;
}

at::Tensor GateUp(const at::Tensor& input, const at::Tensor& packed, int64_t mode)
{
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                input.device() == packed.device(), "input and weight must share one NPU");
    TORCH_CHECK(input.dim() == 2 && input.size(0) > 0 && input.size(0) <= 1595 &&
                input.size(1) == Hidden && input.is_contiguous() && input.storage_offset() == 0 &&
                input.scalar_type() == at::kBFloat16, "expected canonical BF16 input");
    TORCH_CHECK(packed.dim() == 2 && packed.size(0) == 2 * Intermediate &&
                packed.size(1) == Hidden && packed.storage_offset() == 0 &&
                packed.scalar_type() == at::kBFloat16, "expected the privately packed BF16 weight");
    TORCH_CHECK(mode == 0 || mode == 1, "expected long tile raw or fused mode");

    const c10::DeviceGuard deviceGuard(input.device());
    const static Api api;
    auto output = at::empty({input.size(0), (mode == 0 ? 2 : 1) * Intermediate}, input.options());
    const std::array<int64_t, 1> inputStorage{
        static_cast<int64_t>(input.storage().nbytes() / sizeof(at::BFloat16))};
    const std::array<int64_t, 1> outputStorage{output.numel()};
    const std::array<int64_t, 4> weightStorage{Hidden / 16, 2 * Intermediate / 16, 16, 16};
    auto x = MakeDescriptor(api, input, ACL_FORMAT_ND, inputStorage);
    auto w = MakeDescriptor(api, packed, ACL_FORMAT_FRACTAL_NZ, weightStorage);
    auto y = MakeDescriptor(api, output, ACL_FORMAT_ND, outputStorage);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto getStatus = api.getWorkspace(x.get(), w.get(), mode, y.get(), &workspaceSize, &executor);
    TORCH_CHECK(getStatus == 0 && executor && workspaceSize <= (64ULL << 20),
                "invalid workspace request, status ", getStatus, ", size ", workspaceSize);
    at::Tensor workspace;
    void* pointer = nullptr;
    if (workspaceSize) {
        workspace = at::empty({static_cast<int64_t>(workspaceSize)}, input.options().dtype(at::kByte));
        pointer = workspace.data_ptr();
    }

    // Match the reference dispatch's stream and allocator lifetime.
    const auto stream = c10_npu::getCurrentNPUStream(input.device().index()).stream();
    const auto launchStatus = api.launch(pointer, workspaceSize, executor, stream);
    TORCH_CHECK(launchStatus == 0, "private ACLNN launch failed: ", launchStatus);
    return output;
}
}  // namespace

TORCH_LIBRARY(clef_gateup_experiment, m)
{
    m.def("run(Tensor input, Tensor pairedWeight, int mode) -> Tensor");
}

TORCH_LIBRARY_IMPL(clef_gateup_experiment, PrivateUse1, m)
{
    m.impl("run", TORCH_FN(GateUp));
}
