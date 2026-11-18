#pragma once
// Standalone CPU/POSIX trace; no Torch, ACL, device calls or synchronization.
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

namespace grouped29_trace {
class FirstCallTrace {
    int fd_ = -1;
    int rank_ = -1;
    void emit(const char* stage, const char* fields) const noexcept {
        struct timespec ts{};
        if (clock_gettime(CLOCK_MONOTONIC, &ts)) { failure(); return; }
        char line[1024];
        const auto ns = static_cast<unsigned long long>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
        const int n = std::snprintf(line, sizeof(line),
            "{\"schema\":\"grouped29-firstcall/1\",\"source\":\"cpp\",\"rank\":%d,"
            "\"pid\":%ld,\"pgid\":%ld,\"monotonic_ns\":%llu,\"stage\":\"%s\"%s}\n",
            rank_, static_cast<long>(getpid()), static_cast<long>(getpgrp()), ns, stage, fields);
        if (n <= 0 || static_cast<size_t>(n) >= sizeof(line)) { failure(); return; }
        ssize_t wrote;
        do { wrote = ::write(fd_, line, static_cast<size_t>(n)); } while (wrote < 0 && errno == EINTR);
        if (wrote != n) failure();
    }
    static void failure() noexcept {
        // Preserve the actual ACL/control-flow result even if diagnostic I/O fails.
        constexpr char msg[] = "grouped29 first-call trace write failed; trace evidence incomplete\n";
        const auto ignored = ::write(STDERR_FILENO, msg, sizeof(msg) - 1);
        (void)ignored;
    }
public:
    explicit FirstCallTrace(bool enabled = true) {
        if (!enabled) return;
        const char* value = std::getenv("CLEF_GROUPED29_TRACE_FD");
        const char* rank = std::getenv("RANK");
        if (!value || !rank || !((rank[0] == '0' || rank[0] == '1') && rank[1] == '\0'))
            throw std::runtime_error("grouped29 missing first-call rank/trace fd");
        char* end = nullptr;
        errno = 0;
        long original = std::strtol(value, &end, 10);
        if (errno || end == value || *end || original < 0 || original > 2147483647L)
            throw std::runtime_error("grouped29 invalid trace fd");
        int flags = fcntl(static_cast<int>(original), F_GETFL);
        struct stat st{};
        if (flags < 0 || !(flags & O_APPEND) || fstat(static_cast<int>(original), &st) || !S_ISREG(st.st_mode))
            throw std::runtime_error("grouped29 trace fd must be an append regular file");
        fd_ = fcntl(static_cast<int>(original), F_DUPFD_CLOEXEC, 0);
        if (fd_ < 0) throw std::runtime_error("grouped29 could not retain trace fd");
        rank_ = rank[0] - '0';
    }
    FirstCallTrace(const FirstCallTrace&) = delete;
    FirstCallTrace& operator=(const FirstCallTrace&) = delete;
    ~FirstCallTrace() { if (fd_ >= 0) (void)::close(fd_); }
    void mark(const char* stage) const noexcept { if (fd_ >= 0) emit(stage, ""); }
    void rc(const char* stage, long long code) const noexcept {
        if (fd_ < 0) return;
        char fields[80];
        std::snprintf(fields, sizeof(fields), ",\"rc\":%lld", code);
        emit(stage, fields);
    }
    void workspace(long long code, uint64_t bytes, bool executor) const noexcept {
        if (fd_ < 0) return;
        char fields[160];
        std::snprintf(fields, sizeof(fields), ",\"rc\":%lld,\"workspace_bytes\":%llu,\"executor_nonnull\":%s",
                      code, static_cast<unsigned long long>(bytes), executor ? "true" : "false");
        emit("workspace_return", fields);
    }
};
}
