// Standalone aclnn launcher for the norm-only ClefArAddRmsNorm variant.
// Runs one [M,N] BF16 case on device 0 and compares against a CPU FP32
// reference of  z = x + residual; rstd = rsqrt(mean(z^2)+eps); out = z*rstd*gamma.
// Build: see runtime/wave3_norm_launcher_build.sh
#include <acl/acl.h>
#include "aclnn_clef_ar_add_rms_norm.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ACL_CHECK(expr)                                                        \
    do {                                                                       \
        aclError _e = (expr);                                                  \
        if (_e != ACL_SUCCESS) {                                               \
            printf("ACL error %d at %s:%d\n", (int)_e, __FILE__, __LINE__);    \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static uint16_t f2bf(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
    return static_cast<uint16_t>(u >> 16);
}
static bool load_file(const char* path, void* dst, size_t bytes) {
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return false;
    const size_t got = std::fread(dst, 1, bytes, f);
    std::fclose(f);
    return got == bytes;
}

static float bf2f(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? std::atoi(argv[1]) : 967;
    const int N = argc > 2 ? std::atoi(argv[2]) : 5120;
    const char* natdir = argc > 3 ? argv[3] : nullptr;
    const float eps = 1e-6f;
    const size_t elems = static_cast<size_t>(M) * N;
    const int mPad = (M + 7) / 8 * 8;
    const size_t bytes = elems * 2;

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    std::vector<uint16_t> hx(elems), hr(elems), hg(N), hout(elems), hro(elems);
    std::vector<float> href(elems), hrst(M), hrst_dev(mPad);
    std::vector<uint16_t> hy_native(elems), hres_native(elems);
    std::vector<float> hrstd_native(mPad);
    bool have_native = false;
    if (natdir != nullptr) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/x.bin", natdir);
        const bool ox = load_file(path, hx.data(), bytes);
        std::snprintf(path, sizeof(path), "%s/r.bin", natdir);
        const bool orr = load_file(path, hr.data(), bytes);
        std::snprintf(path, sizeof(path), "%s/g.bin", natdir);
        const bool og = load_file(path, hg.data(), N * 2);
        std::snprintf(path, sizeof(path), "%s/y.bin", natdir);
        const bool oy = load_file(path, hy_native.data(), bytes);
        std::snprintf(path, sizeof(path), "%s/res.bin", natdir);
        const bool ores = load_file(path, hres_native.data(), bytes);
        std::snprintf(path, sizeof(path), "%s/rstd.bin", natdir);
        const bool ors = load_file(path, hrstd_native.data(), static_cast<size_t>(M) * 4);
        have_native = ox && orr && og && oy && ores && ors;
        printf("native inputs loaded=%d (x=%d r=%d g=%d y=%d res=%d rstd=%d)\n",
               (int)have_native, (int)ox, (int)orr, (int)og, (int)oy, (int)ores, (int)ors);
        if (!have_native) return 4;
    }
    if (!have_native)
    for (size_t i = 0; i < elems; ++i) {
        hx[i] = f2bf(std::sin(static_cast<double>(i) * 1e-3) * 0.75f);
        hr[i] = f2bf(std::cos(static_cast<double>(i) * 7e-4) * 0.5f);
    }
    if (!have_native)
    for (int i = 0; i < N; ++i) hg[i] = f2bf(0.5f + 0.5f * std::sin(static_cast<double>(i) * 1e-2));
    if (!have_native)
    for (int row = 0; row < M; ++row) {
        double q = 0.0;
        for (int c = 0; c < N; ++c) {
            const size_t i = static_cast<size_t>(row) * N + c;
            const float z = bf2f(hx[i]) + bf2f(hr[i]);
            q += static_cast<double>(z) * z;
        }
        const float rstd = 1.0f / std::sqrt(static_cast<float>(q / N) + eps);
        hrst[row] = rstd;
        for (int c = 0; c < N; ++c) {
            const size_t i = static_cast<size_t>(row) * N + c;
            const float z = bf2f(hx[i]) + bf2f(hr[i]);
            href[i] = z * rstd * bf2f(hg[c]);
        }
    }

    void *dx, *dr, *dg, *dout, *dro, *drstd;
    ACL_CHECK(aclrtMalloc(&dx, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dg, N * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dout, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dro, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&drstd, static_cast<size_t>(mPad) * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(dx, bytes, hx.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dr, bytes, hr.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dg, N * 2, hg.data(), N * 2, ACL_MEMCPY_HOST_TO_DEVICE));

    const int64_t dims[2] = {M, N};
    const int64_t strides[2] = {N, 1};
    const int64_t rdims[1] = {M};
    const int64_t rstore[1] = {mPad};
    const int64_t rstrides[1] = {1};
    aclTensor* tx = aclCreateTensor(dims, 2, ACL_BF16, strides, 0, ACL_FORMAT_ND, dims, 2, dx);
    aclTensor* tr = aclCreateTensor(dims, 2, ACL_BF16, strides, 0, ACL_FORMAT_ND, dims, 2, dr);
    aclTensor* tg = aclCreateTensor(rdims, 1, ACL_BF16, rstrides, 0, ACL_FORMAT_ND, rdims, 1, dg);
    aclTensor* to = aclCreateTensor(dims, 2, ACL_BF16, strides, 0, ACL_FORMAT_ND, dims, 2, dout);
    aclTensor* tro = aclCreateTensor(dims, 2, ACL_BF16, strides, 0, ACL_FORMAT_ND, dims, 2, dro);
    aclTensor* trs = aclCreateTensor(rdims, 1, ACL_FLOAT, rstrides, 0, ACL_FORMAT_ND, rstore, 1, drstd);

    uint64_t workspace = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus st = aclnnClefArAddRmsNormGetWorkspaceSize(
        tx, tr, tg, static_cast<double>(eps), 1, to, tro, trs, &workspace, &executor);
    printf("GetWorkspaceSize status=%d workspace=%llu executor=%p\n", (int)st,
           (unsigned long long)workspace, (void*)executor);
    if (st != 0 || executor == nullptr) return 2;
    void* ws = nullptr;
    if (workspace > 0) ACL_CHECK(aclrtMalloc(&ws, workspace, ACL_MEM_MALLOC_HUGE_FIRST));
    st = aclnnClefArAddRmsNorm(ws, workspace, executor, stream);
    printf("exec status=%d\n", (int)st);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(hout.data(), bytes, dout, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hro.data(), bytes, dro, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(hrst_dev.data(), static_cast<size_t>(mPad) * 4, drstd,
                          static_cast<size_t>(mPad) * 4, ACL_MEMCPY_DEVICE_TO_HOST));

    double max_out = 0.0, max_res = 0.0, max_rstd = 0.0;
    size_t bad_out = 0;
    for (size_t i = 0; i < elems; ++i) {
        const double d = std::fabs(static_cast<double>(bf2f(hout[i])) - href[i]);
        if (d > max_out) max_out = d;
        if (d > 0.01) ++bad_out;
        const double dz = std::fabs(static_cast<double>(bf2f(hro[i])) -
                                    (static_cast<double>(bf2f(hx[i])) + bf2f(hr[i])));
        if (dz > max_res) max_res = dz;
    }
    for (int row = 0; row < M; ++row) {
        const double d = std::fabs(static_cast<double>(hrst_dev[row]) - hrst[row]);
        if (d > max_rstd) max_rstd = d;
    }
    size_t nan_out = 0, inf_out = 0, first_bad = SIZE_MAX;
    double first_bad_val = 0.0;
    for (size_t i = 0; i < elems; ++i) {
        const double v = bf2f(hout[i]);
        if (std::isnan(v)) ++nan_out;
        else if (std::isinf(v)) ++inf_out;
        if (first_bad == SIZE_MAX && std::fabs(v) > 100.0) { first_bad = i; first_bad_val = v; }
    }
    printf("nan_out=%zu inf_out=%zu first_bad_idx=%zu row=%zu col=%zu val=%g\n",
           nan_out, inf_out, first_bad, first_bad == SIZE_MAX ? 0 : first_bad / N,
           first_bad == SIZE_MAX ? 0 : first_bad % N, first_bad_val);
    printf("raw: out[0..3]=%g,%g,%g,%g rstd[0..3]=%g,%g,%g,%g\n",
           bf2f(hout[0]), bf2f(hout[1]), bf2f(hout[2]), bf2f(hout[3]),
           hrst_dev[0], hrst_dev[1], hrst_dev[2], hrst_dev[3]);
    printf("M=%d N=%d ref=%s max_abs_out=%.6g max_abs_residual=%.6g max_abs_rstd=%.6g "
           "out_gt_0.01=%zu/%zu\n", M, N, have_native ? "native" : "cpu_fp32_on_bf16_inputs",
           max_out, max_res, max_rstd, bad_out, elems);
    if (have_native) {
        double dy = 0.0, dres = 0.0, drs = 0.0;
        size_t bad = 0;
        for (size_t i = 0; i < elems; ++i) {
            const double a = std::fabs(static_cast<double>(bf2f(hout[i])) - bf2f(hy_native[i]));
            const double b = std::fabs(static_cast<double>(bf2f(hro[i])) - bf2f(hres_native[i]));
            if (a > dy) dy = a;
            if (b > dres) dres = b;
            if (a > 0.01) ++bad;
        }
        for (int row = 0; row < M; ++row) {
            const double d = std::fabs(static_cast<double>(hrst_dev[row]) - hrstd_native[row]);
            if (d > drs) drs = d;
        }
        printf("vs_native: max_abs_Y=%.6g max_abs_residual_sum=%.6g max_abs_rstd=%.6g "
               "Y_gt_0.01=%zu/%zu\n", dy, dres, drs, bad, elems);
        printf("Y ours : ");
        for (int k = 0; k < 8; ++k) printf("%.5g ", bf2f(hout[k]));
        printf("\nY nativ: ");
        for (int k = 0; k < 8; ++k) printf("%.5g ", bf2f(hy_native[k]));
        printf("\nz*rstd*g in: ");
        for (int k = 0; k < 8; ++k) {
            const float zz = bf2f(hx[k]) + bf2f(hr[k]);
            printf("%.5g ", zz * hrst_dev[0] * bf2f(hg[k]));
        }
        printf("\n");
    }

    aclDestroyTensor(tx); aclDestroyTensor(tr); aclDestroyTensor(tg);
    aclDestroyTensor(to); aclDestroyTensor(tro); aclDestroyTensor(trs);
    if (ws != nullptr) aclrtFree(ws);
    aclrtFree(dx); aclrtFree(dr); aclrtFree(dg); aclrtFree(dout);
    aclrtFree(dro); aclrtFree(drstd);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return (max_out <= 0.01 && max_rstd <= 1e-3) ? 0 : 3;
}
