#pragma once
// Rich, unbuffered per-rank trace for the Grouped MMAR probe.
// CPU/POSIX only: no Torch, ACL or device calls; CLOCK_MONOTONIC is shared
// with the Python runner.
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace grouped_mmar_trace {
class ProbeTrace {
    int fd_ = -1;
    int rank_ = -1;
    void emit(const char* stage, const char* fields) const noexcept {
        struct timespec ts{};
        if (clock_gettime(CLOCK_MONOTONIC, &ts)) return;
        const auto ns = static_cast<unsigned long long>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
        const long tid = static_cast<long>(::syscall(SYS_gettid));
        char line[1400];
        const int n = std::snprintf(
            line, sizeof(line),
            "{\"schema\":\"grouped-mmar-probe/1\",\"source\":\"cpp\",\"rank\":%d,"
            "\"pid\":%ld,\"pgid\":%ld,\"tid\":%ld,\"monotonic_ns\":%llu,"
            "\"stage\":\"%s\"%s}\n",
            rank_, static_cast<long>(getpid()), static_cast<long>(getpgrp()), tid, ns,
            stage, fields);
        if (n <= 0 || static_cast<size_t>(n) >= sizeof(line)) return;
        ssize_t wrote;
        do {
            wrote = ::write(fd_, line, static_cast<size_t>(n));
        } while (wrote < 0 && errno == EINTR);
    }

public:
    explicit ProbeTrace(bool enabled = true) {
        if (!enabled) return;
        const char* value = std::getenv("CLEF_GROUPED_MMAR_TRACE_FD");
        if (!value) value = std::getenv("CLEF_GROUPED29_TRACE_FD");
        const char* rank = std::getenv("RANK");
        if (!value || !rank || !((rank[0] == '0' || rank[0] == '1') && rank[1] == '\0'))
            throw std::runtime_error("grouped-mmar missing RANK/TRACE_FD");
        char* end = nullptr;
        errno = 0;
        long original = std::strtol(value, &end, 10);
        if (errno || end == value || *end || original < 0 || original > 2147483647L)
            throw std::runtime_error("grouped-mmar invalid trace fd");
        int flags = fcntl(static_cast<int>(original), F_GETFL);
        struct stat st{};
        if (flags < 0 || !(flags & O_APPEND) || fstat(static_cast<int>(original), &st) ||
            !S_ISREG(st.st_mode))
            throw std::runtime_error("grouped-mmar trace fd must be append regular file");
        fd_ = fcntl(static_cast<int>(original), F_DUPFD_CLOEXEC, 0);
        if (fd_ < 0) throw std::runtime_error("grouped-mmar cannot retain trace fd");
        rank_ = rank[0] - '0';
    }
    ~ProbeTrace() { if (fd_ >= 0) (void)::close(fd_); }
    ProbeTrace(const ProbeTrace&) = delete;
    ProbeTrace& operator=(const ProbeTrace&) = delete;
    bool enabled() const noexcept { return fd_ >= 0; }
    void mark(const char* stage, const char* fields = "") const noexcept {
        if (fd_ >= 0) emit(stage, fields);
    }
};
}  // namespace grouped_mmar_trace
