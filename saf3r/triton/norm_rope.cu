#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

using bf16 = __nv_bfloat16;

__device__ __forceinline__ void normalize_rotate(
    const bf16* input, const bf16* weight, const bf16* bias, bf16* output,
    const float* cosine, const float* sine, float epsilon, int lane,
    unsigned mask) {
    float values[4];
    float mean = 0.f, m2 = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        values[i] = __bfloat162float(input[4 * lane + i]);
        float delta = values[i] - mean;
        mean = __fmaf_rn(delta, 1.f / (i + 1), mean);
        m2 = __fmaf_rn(delta, values[i] - mean, m2);
    }
    float count = 4.f;
    #pragma unroll
    for (int step = 8; step > 0; step >>= 1) {
        float peer_mean = __shfl_down_sync(mask, mean, step, 16);
        float peer_m2 = __shfl_down_sync(mask, m2, step, 16);
        float delta = mean - peer_mean;
        mean = __fmaf_rn(.5f, peer_mean, .5f * mean);
        m2 = __fmaf_rn(delta * delta * count, .5f, m2 + peer_m2);
        count *= 2.f;
    }
    mean = __shfl_sync(mask, mean, 0, 16);
    m2 = __shfl_sync(mask, m2, 0, 16);
    float inv = rsqrtf(m2 * (1.f / 64) + epsilon);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int feature = 4 * lane + i;
        float current = __fmaf_rn((values[i] - mean) * inv,
                                 __bfloat162float(weight[feature]),
                                 __bfloat162float(bias[feature]));
        float partner = __shfl_xor_sync(mask, current, 4, 16);
        float rotated = (lane & 4) ? partner : -partner;
        int local = feature & 31;
        output[feature] = __float2bfloat16(__fadd_rn(
            __fmul_rn(current, cosine[local]), __fmul_rn(rotated, sine[local])));
    }
}

__global__ void norm_rope(
    const bf16* q, const bf16* k, const bf16* v,
    const bf16* qw, const bf16* qb, const bf16* kw, const bf16* kb,
    const int64_t* pos, const float* cosine, const float* sine,
    bf16* qo, bf16* ko, bf16* vo, int n, float eq, float ek,
    int qh, int qt, int kh, int kt, int vh, int vt) {
    int token = blockIdx.x * 8 + threadIdx.x / 16;
    int head = blockIdx.y;
    int lane = threadIdx.x & 15;
    unsigned mask = __ballot_sync(0xffffffff, token < n);
    if (token >= n) return;
    int pos_index = pos[2 * token + (lane >= 8)];
    const float* cos_row = cosine + 32 * pos_index;
    const float* sin_row = sine + 32 * pos_index;
    int64_t dst = (static_cast<int64_t>(head) * n + token) * 64;
    normalize_rotate(q + int64_t(head) * qh + int64_t(token) * qt,
                     qw, qb, qo + dst, cos_row, sin_row, eq, lane, mask);
    normalize_rotate(k + int64_t(head) * kh + int64_t(token) * kt,
                     kw, kb, ko + dst, cos_row, sin_row, ek, lane, mask);
    int64_t src_v = int64_t(head) * vh + int64_t(token) * vt + 4 * lane;
    *reinterpret_cast<uint64_t*>(vo + dst + 4 * lane) =
        *reinterpret_cast<const uint64_t*>(v + src_v);
}

extern "C" int launch_norm_rope(
    const bf16* q, const bf16* k, const bf16* v,
    const bf16* qw, const bf16* qb, const bf16* kw, const bf16* kb,
    const int64_t* pos, const float* cosine, const float* sine,
    bf16* qo, bf16* ko, bf16* vo, int n, int heads, float eq, float ek,
    int qh, int qt, int kh, int kt, int vh, int vt, cudaStream_t stream) {
    norm_rope<<<dim3((n + 7) / 8, heads), 128, 0, stream>>>(
        q, k, v, qw, qb, kw, kb, pos, cosine, sine, qo, ko, vo,
        n, eq, ek, qh, qt, kh, kt, vh, vt);
    return cudaGetLastError();
}
