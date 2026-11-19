// Minimal native GroupedMatMulAllReduce probe with single-variable lifetime,
// thread/context/stream and workspace experiments.  AOT only; the runner loads
// it on device.  No service, no norm, no f810 restore.
#include "probe_trace.h"
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
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace {
constexpr const char* OPAPI =
    "/usr/local/Ascend/cann-9.2.1/aarch64-linux/lib64/libopapi.so";
constexpr const char* META =
    "/usr/local/Ascend/cann-9.2.1/aarch64-linux/lib64/libnnopbase.so";
constexpr int KEEP_GROUP = 1;
constexpr int KEEP_WORKSPACE = 2;
constexpr int KEEP_DESCRIPTORS = 4;
constexpr int KEEP_ALL = 8;
constexpr int DIRECT_LAUNCH = 16;
constexpr int REFETCH_STREAM = 32;
constexpr int MEMSET_WORKSPACE = 64;

std::string canonical(const char* path) {
    std::unique_ptr<char, decltype(&std::free)> p(realpath(path, nullptr), std::free);
    TORCH_CHECK(p, "grouped-mmar realpath failed: ", path);
    return p.get();
}
template <class F>
F symbol(void* lib, const char* name, const char* expected) {
    dlerror();
    void* p = dlsym(lib, name);
    const char* error = dlerror();
    TORCH_CHECK(p && !error, "grouped-mmar missing symbol: ", name, " ", error ? error : "");
    Dl_info info{};
    TORCH_CHECK(dladdr(p, &info) && info.dli_fname && canonical(info.dli_fname) == canonical(expected),
                "grouped-mmar unexpected symbol provider: ", name);
    return reinterpret_cast<F>(p);
}
struct Api {
    void* lib = nullptr;
    void* meta = nullptr;
    decltype(&aclnnGroupedMatMulAllReduceGetWorkspaceSize) prepare = nullptr;
    decltype(&aclnnGroupedMatMulAllReduce) launch = nullptr;
    decltype(&aclCreateTensor) create_tensor = nullptr;
    decltype(&aclCreateTensorList) create_list = nullptr;
    decltype(&aclDestroyTensor) destroy_tensor = nullptr;
    decltype(&aclDestroyTensorList) destroy_list = nullptr;
    Api() {
        lib = dlopen(OPAPI, RTLD_NOW | RTLD_LOCAL);
        TORCH_CHECK(lib, "grouped-mmar opapi dlopen: ", dlerror());
        meta = dlopen(META, RTLD_NOW | RTLD_LOCAL);
        TORCH_CHECK(meta, "grouped-mmar metadata dlopen: ", dlerror());
        prepare = symbol<decltype(prepare)>(lib, "aclnnGroupedMatMulAllReduceGetWorkspaceSize", OPAPI);
        launch = symbol<decltype(launch)>(lib, "aclnnGroupedMatMulAllReduce", OPAPI);
        create_tensor = symbol<decltype(create_tensor)>(meta, "aclCreateTensor", META);
        create_list = symbol<decltype(create_list)>(meta, "aclCreateTensorList", META);
        destroy_tensor = symbol<decltype(destroy_tensor)>(meta, "aclDestroyTensor", META);
        destroy_list = symbol<decltype(destroy_list)>(meta, "aclDestroyTensorList", META);
        // Keep providers mapped until process exit: queued launches may still
        // resolve their symbols after this translation unit is unloaded.
    }
};
Api& api() { static Api value; return value; }

struct Lists {
    Api& a;
    std::vector<aclTensorList*> lists;
    bool closed = false;
    explicit Lists(Api& value) : a(value) {}
    int close() noexcept {
        if (closed) return 0;
        closed = true;
        int first_error = 0;
        for (auto* list : lists) {
            int rc = a.destroy_list(list);  // Owns/destroys its member descriptors.
            if (rc && !first_error) first_error = rc;
        }
        lists.clear();
        return first_error;
    }
    ~Lists() { (void)close(); }
    aclTensorList* add(const at::Tensor& t, aclDataType dtype) {
        const auto* storage =
            static_cast<const torch_npu::NPUStorageImpl*>(t.storage().unsafeGetStorageImpl());
        const auto& desc = storage->npu_desc_;
        TORCH_CHECK(static_cast<aclFormat>(desc.npu_format_) == ACL_FORMAT_ND,
                    "grouped-mmar requires actual ND storage, got ",
                    static_cast<int>(desc.npu_format_));
        std::vector<int64_t> sizes(t.sizes().begin(), t.sizes().end());
        std::vector<int64_t> strides(t.strides().begin(), t.strides().end());
        std::vector<int64_t> storage_sizes(desc.storage_sizes_.begin(), desc.storage_sizes_.end());
        auto* tensor = a.create_tensor(sizes.data(), sizes.size(), dtype, strides.data(),
                                       t.storage_offset(), ACL_FORMAT_ND, storage_sizes.data(),
                                       storage_sizes.size(), const_cast<void*>(t.storage().data()));
        TORCH_CHECK(tensor, "grouped-mmar aclCreateTensor returned null");
        const aclTensor* members[] = {tensor};
        auto* list = a.create_list(members, 1);
        if (!list) {
            const int rc = a.destroy_tensor(tensor);
            TORCH_CHECK(false, "grouped-mmar aclCreateTensorList returned null, cleanup rc=", rc);
        }
        lists.push_back(list);  // Exactly one owner; no standalone tensor destroy.
        return list;
    }
};

struct RunState {
    int flags = 0;
    int id = 0;
    // Keepalive fields are populated ONLY for the flag under test.  A H1 arm
    // must not accidentally keep the workspace/lists alive, and a H2 arm must
    // not keep the group string/descriptors alive.
    std::shared_ptr<std::string> group_keep;
    const char* group_ptr = nullptr;
    at::Tensor keep_workspace;
    std::shared_ptr<Lists> keep_lists;
    at::Tensor keep_x, keep_w, keep_bias, keep_y;
    // Logging-only addresses; never ownership.
    const void* workspace_ptr = nullptr;
    const void* lists_ptr = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t bytes = 0;
    aclrtStream stream = nullptr;
    std::shared_ptr<grouped_mmar_trace::ProbeTrace> trace;
};

std::mutex g_mutex;
std::map<int, std::shared_ptr<RunState>> g_states;
int g_next_id = 1;

bool keep_group(int flags) { return flags & (KEEP_GROUP | KEEP_ALL); }
bool keep_workspace(int flags) { return flags & (KEEP_WORKSPACE | KEEP_ALL); }
bool keep_descriptors(int flags) { return flags & (KEEP_DESCRIPTORS | KEEP_ALL); }

std::string hex(const void* p) {
    char b[32];
    std::snprintf(b, sizeof(b), "0x%llx", static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(p)));
    return b;
}
int current_device() {
    int d = -1;
    (void)aclrtGetDevice(&d);
    return d;
}
aclrtContext current_context() {
    aclrtContext c = nullptr;
    (void)aclrtGetCurrentContext(&c);
    return c;
}
std::shared_ptr<grouped_mmar_trace::ProbeTrace> make_trace() {
    const char* fd = std::getenv("CLEF_GROUPED_MMAR_TRACE_FD");
    if (!fd) fd = std::getenv("CLEF_GROUPED29_TRACE_FD");
    if (!fd) return nullptr;
    return std::make_shared<grouped_mmar_trace::ProbeTrace>();
}
std::string fields(const RunState& s, const char* phase, const void* group_obj,
                   const void* group_data, size_t group_size, size_t group_capacity,
                   aclrtStream stream) {
    char b[640];
    std::snprintf(
        b, sizeof(b),
        ",\"phase\":\"%s\",\"flags\":%d,\"device\":%d,\"context\":\"%s\","
        "\"stream\":\"%s\",\"group_object\":\"%s\",\"group_data\":\"%s\","
        "\"group_size\":%zu,\"group_capacity\":%zu,\"group_data_minus_object\":%lld,"
        "\"workspace\":\"%s\",\"workspace_bytes\":%llu,\"descriptors\":\"%s\","
        "\"executor\":\"%s\"",
        phase, s.flags, current_device(), hex(current_context()).c_str(), hex(stream).c_str(),
        hex(group_obj).c_str(), hex(group_data).c_str(), group_size, group_capacity,
        static_cast<long long>(reinterpret_cast<const char*>(group_data) -
                               reinterpret_cast<const char*>(group_obj)),
        hex(s.workspace_ptr).c_str(),
        static_cast<unsigned long long>(s.bytes),
        hex(s.lists_ptr).c_str(),
        hex(static_cast<const void*>(s.executor)).c_str());
    return b;
}
}  // namespace

std::tuple<at::Tensor, int64_t, int64_t> grouped_mmar_run(
    const at::Tensor& x, const at::Tensor& weight_kn, const at::Tensor& bias,
    const std::string& group, int64_t flags_value) {
    const int flags = static_cast<int>(flags_value);
    TORCH_CHECK(flags >= 0 && flags <= 127, "grouped-mmar invalid flags: ", flags);
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "grouped-mmar requires NPU input");
    c10_npu::NPUGuard guard(x.device());
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 16384,
                "grouped-mmar x must be [M,K]");
    const auto rows = x.size(0), k = x.size(1), n = weight_kn.size(1);
    TORCH_CHECK(weight_kn.dim() == 2 && weight_kn.size(0) == k && n > 0,
                "grouped-mmar weight must be [K,N] matching x");
    for (const auto* t : {&x, &weight_kn}) {
        TORCH_CHECK(t->device() == x.device() && t->scalar_type() == at::kBFloat16 &&
                    t->is_contiguous() && t->storage_offset() == 0,
                    "grouped-mmar BF16/input layout mismatch");
    }
    TORCH_CHECK(bias.device() == x.device() && bias.scalar_type() == at::kFloat &&
                bias.dim() == 1 && bias.size(0) == n && bias.is_contiguous() &&
                bias.storage_offset() == 0, "grouped-mmar requires contiguous FP32 bias [N]");
    TORCH_CHECK(!group.empty() && group.find('\0') == std::string::npos,
                "grouped-mmar invalid HCCL group");

    Api& a = api();
    auto state = std::make_shared<RunState>();
    state->flags = flags;
    state->trace = make_trace();
    auto y = at::empty({rows, n}, x.options());

    // H1: the keepalive object must be created BEFORE prepare and its own
    // c_str() passed; a copy made after prepare cannot preserve that pointer.
    std::string group_local(group);
    const void* group_object = nullptr;
    const char* group_pointer = nullptr;
    size_t group_capacity = 0;
    if (keep_group(flags)) {
        state->group_keep = std::make_shared<std::string>(group);
        group_object = state->group_keep.get();
        group_pointer = state->group_keep->c_str();
        group_capacity = state->group_keep->capacity();
    } else {
        group_object = &group_local;
        group_pointer = group_local.c_str();
        group_capacity = group_local.capacity();
    }
    state->group_ptr = group_pointer;

    auto lists = std::make_shared<Lists>(a);
    state->lists_ptr = lists.get();
    if (keep_descriptors(flags)) state->keep_lists = lists;
    auto* tx = lists->add(x, ACL_BF16);
    auto* tw = lists->add(weight_kn, ACL_BF16);
    auto* tb = lists->add(bias, ACL_FLOAT);
    auto* ty = lists->add(y, ACL_BF16);
    if (state->trace)
        state->trace->mark("descriptors_ready",
                           fields(*state, "descriptors", group_object, group_pointer,
                                  group.size(), group_capacity, nullptr).c_str());

    aclOpExecutor* executor = nullptr;
    uint64_t bytes = 0;
    const auto prepare_rc = a.prepare(tx, tw, tb, nullptr, 0, group_pointer, "sum", 0, 1,
                                      ty, &bytes, &executor);
    state->executor = executor;
    state->bytes = bytes;
    state->stream = c10_npu::getCurrentNPUStream().stream();
    if (state->trace) {
        char extra[160];
        std::snprintf(extra, sizeof(extra), ",\"prepare_rc\":%d,\"executor_nonnull\":%s",
                      prepare_rc, executor ? "true" : "false");
        state->trace->mark("prepare_return",
                           (fields(*state, "prepare", group_object, group_pointer, group.size(),
                                   group_capacity, state->stream) + extra).c_str());
    }
    if (prepare_rc != 0 || !executor) {
        const std::string error = aclGetRecentErrMsg() ? aclGetRecentErrMsg() : "";
        const int cleanup = lists->close();
        TORCH_CHECK(false, "grouped-mmar prepare rc=", prepare_rc, " executor=", executor,
                    " cleanup rc=", cleanup, " ", error);
    }
    TORCH_CHECK(bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "grouped-mmar workspace overflow");
    auto workspace = at::empty({static_cast<int64_t>(bytes)}, x.options().dtype(at::kByte));
    state->workspace_ptr = workspace.data_ptr();
    if (keep_workspace(flags)) state->keep_workspace = workspace;
    if (flags & KEEP_ALL) {
        state->keep_x = x;
        state->keep_w = weight_kn;
        state->keep_bias = bias;
        state->keep_y = y;
        state->keep_workspace = workspace;
        state->keep_lists = lists;
    }

    int id = 0;
    if (keep_group(flags) || keep_workspace(flags) || keep_descriptors(flags)) {
        std::lock_guard<std::mutex> lock(g_mutex);
        id = g_next_id++;
        state->id = id;
        g_states[id] = state;  // Retain until release() after sync is confirmed.
    }

    auto body = [state, a, workspace, lists, y, prepare_rc, group_object, group_pointer,
                 group_capacity, rows, k, n]() -> int {
        char extra[96];
        std::snprintf(extra, sizeof(extra), ",\"rows\":%lld,\"k\":%lld,\"n\":%lld",
                      static_cast<long long>(rows), static_cast<long long>(k),
                      static_cast<long long>(n));
        if (state->trace)
            state->trace->mark("handler_enter",
                               (fields(*state, "handler", group_object, group_pointer, 0,
                                       group_capacity, state->stream) + extra).c_str());
        aclrtStream use_stream = state->stream;
        if (state->flags & REFETCH_STREAM) use_stream = c10_npu::getCurrentNPUStream().stream();
        if (state->trace) {
            char s[160];
            std::snprintf(s, sizeof(s), ",\"prepare_stream\":\"%s\",\"handler_stream\":\"%s\"",
                          hex(state->stream).c_str(), hex(use_stream).c_str());
            state->trace->mark("stream_compare", s);
        }
        int memset_rc = 0;
        if (state->flags & MEMSET_WORKSPACE) {
            memset_rc = aclrtMemsetAsync(workspace.data_ptr(), state->bytes, 0,
                                         state->bytes, use_stream);
            if (state->trace) {
                char s[80];
                std::snprintf(s, sizeof(s), ",\"memset_rc\":%d", memset_rc);
                state->trace->mark("memset_return", s);
            }
        }
        if (state->trace) state->trace->mark("launch_enter");
        const int launch_rc = a.launch(state->bytes ? workspace.data_ptr() : nullptr,
                                       state->bytes, state->executor, use_stream);
        if (state->trace) {
            char s[80];
            std::snprintf(s, sizeof(s), ",\"launch_rc\":%d", launch_rc);
            state->trace->mark("launch_return", s);
        }
        int sync_rc = 0;
        if (state->flags & DIRECT_LAUNCH) {
            if (state->trace) state->trace->mark("sync_enter");
            sync_rc = aclrtSynchronizeStream(use_stream);
            if (state->trace) {
                char s[80];
                std::snprintf(s, sizeof(s), ",\"sync_rc\":%d", sync_rc);
                state->trace->mark("sync_return", s);
            }
        }
        int cleanup_rc = 0;
        if (!keep_descriptors(state->flags)) cleanup_rc = lists->close();
        if (state->trace) {
            char s[96];
            std::snprintf(s, sizeof(s), ",\"cleanup_rc\":%d,\"sync_rc\":%d", cleanup_rc, sync_rc);
            state->trace->mark("handler_return", s);
        }
        const std::string error = (launch_rc || sync_rc || memset_rc || cleanup_rc)
                                      ? (aclGetRecentErrMsg() ? aclGetRecentErrMsg() : "")
                                      : "";
        TORCH_CHECK(launch_rc == 0 && sync_rc == 0 && memset_rc == 0 && cleanup_rc == 0,
                    "grouped-mmar handler launch=", launch_rc, " sync=", sync_rc,
                    " memset=", memset_rc, " cleanup=", cleanup_rc, " ", error);
        return 0;
    };

    if (flags & DIRECT_LAUNCH) {
        body();
    } else {
        at_npu::native::OpCommand cmd;
        cmd.Name("ClefGroupedMmarProbe");
        cmd.SetCustomHandler([body]() -> int { return body(); });
        if (state->trace) state->trace->mark("dispatch_enter");
        cmd.Run();
        if (state->trace) state->trace->mark("dispatch_return");
    }
    return {y, static_cast<int64_t>(state->bytes), static_cast<int64_t>(id)};
}

int64_t grouped_mmar_release(int64_t id_value) {
    const int id = static_cast<int>(id_value);
    std::shared_ptr<RunState> state;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_states.find(id);
        if (it == g_states.end()) return 0;
        state = it->second;
        g_states.erase(it);
    }
    if (state->trace) {
        char s[96];
        std::snprintf(s, sizeof(s), ",\"run_id\":%d,\"flags\":%d", id, state->flags);
        state->trace->mark("release_enter", s);
    }
    const int rc = state->keep_lists ? state->keep_lists->close() : 0;
    if (state->trace) {
        char s[96];
        std::snprintf(s, sizeof(s), ",\"run_id\":%d,\"cleanup_rc\":%d", id, rc);
        state->trace->mark("release_return", s);
    }
    state->keep_workspace = at::Tensor();
    state->keep_x = at::Tensor();
    state->keep_w = at::Tensor();
    state->keep_bias = at::Tensor();
    state->keep_y = at::Tensor();
    state->keep_lists.reset();
    state->group_keep.reset();
    state->executor = nullptr;
    return rc;
}

std::string grouped_mmar_provider() {
    (void)api();
    return std::string("prepare/launch=") + canonical(OPAPI) + ";metadata=" + canonical(META);
}

TORCH_LIBRARY(clef_mmar_probe, m) {
    m.def("run(Tensor x, Tensor weight_kn, Tensor bias, str group, int flags) -> (Tensor, int, int)");
    m.def("release(int id) -> int");
    m.def("provider() -> str", &grouped_mmar_provider);
}
TORCH_LIBRARY_IMPL(clef_mmar_probe, PrivateUse1, m) { m.impl("run", &grouped_mmar_run); }
