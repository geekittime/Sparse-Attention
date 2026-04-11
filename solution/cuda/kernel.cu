/*
 * Fused DeltaNet Recurrent State Update - CUDA
 *
 * Best-of-breed design combining all proven optimizations:
 *   1. float4 (128-bit) vectorized state load/store
 *   2. uint2 (64-bit) vectorized q/k/v bf16 loads
 *   3. 4 warps per block — higher occupancy without smem overhead
 *   4. No shared memory, no __syncthreads — warps fully independent
 *   5. Each warp loads its own q/k into registers (256B bf16 = trivial)
 *   6. Grid: (NV_BLOCKS, B*HV) — maximum block-level parallelism
 *
 * State layout: [B, HV, V, K] (k-last, K contiguous) — f32
 *
 * Grid: (V/(4*BV), B*HV) = (8, B*8)
 * Block: 128 threads (4 warps x 32 lanes)
 * Each warp: BV=4 v-rows, KPT=4 floats/thread
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

// TVM FFI headers
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/error.h>

constexpr int K_DIM = 128;
constexpr int V_DIM = 128;
constexpr int WARP_SIZE = 32;
constexpr int BV = 4;
constexpr int NUM_WARPS = 4;
constexpr int BV_TOTAL = NUM_WARPS * BV;  // 16 v-rows per block
constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;  // 128 threads
constexpr int KPT = K_DIM / WARP_SIZE;   // 4

__device__ __forceinline__ float softplus_f(float x) {
    return (x > 20.0f) ? x : __logf(1.0f + __expf(x));
}

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void deltanet_recurrent_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_in,
    const __nv_bfloat16* __restrict__ V_in,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ A,
    const float* __restrict__ Dt_bias,
    const __nv_bfloat16* __restrict__ B_gate,
    const float* __restrict__ S,
    float* __restrict__ New_S,
    __nv_bfloat16* __restrict__ O,
    float scale,
    int H,
    int HV
) {
    const int i_vb = blockIdx.x;
    const int bh = blockIdx.y;
    const int batch_id = bh / HV;
    const int v_head_id = bh % HV;
    const int head_id = v_head_id / (HV / H);

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // ===== Gate/beta computation =====
    const float a_val = __bfloat162float(A[batch_id * HV + v_head_id]);
    const float dt_bias_val = Dt_bias[v_head_id];
    const float a_log_val = A_log[v_head_id];
    const float b_val = __bfloat162float(B_gate[batch_id * HV + v_head_id]);

    const float x = a_val + dt_bias_val;
    const float sp = softplus_f(x);
    const float log_decay = -__expf(a_log_val) * sp;
    const float decay = __expf(log_decay);
    const float beta = sigmoid_f(b_val);

    // ===== Load q, k per-warp via uint2 (64-bit) =====
    const __nv_bfloat16* q_base = Q + batch_id * (H * K_DIM) + head_id * K_DIM;
    const __nv_bfloat16* k_base = K_in + batch_id * (H * K_DIM) + head_id * K_DIM;

    float r_q[KPT], r_k[KPT];
    {
        const uint2* q_u2 = reinterpret_cast<const uint2*>(q_base);
        const uint2* k_u2 = reinterpret_cast<const uint2*>(k_base);
        uint2 q_packed = q_u2[lane_id];
        uint2 k_packed = k_u2[lane_id];
        const __nv_bfloat162* q_bf2 = reinterpret_cast<const __nv_bfloat162*>(&q_packed);
        const __nv_bfloat162* k_bf2 = reinterpret_cast<const __nv_bfloat162*>(&k_packed);
        r_q[0] = __bfloat162float(q_bf2[0].x) * scale;
        r_q[1] = __bfloat162float(q_bf2[0].y) * scale;
        r_q[2] = __bfloat162float(q_bf2[1].x) * scale;
        r_q[3] = __bfloat162float(q_bf2[1].y) * scale;
        r_k[0] = __bfloat162float(k_bf2[0].x);
        r_k[1] = __bfloat162float(k_bf2[0].y);
        r_k[2] = __bfloat162float(k_bf2[1].x);
        r_k[3] = __bfloat162float(k_bf2[1].y);
    }

    // ===== Pointers =====
    const int s_head_offset = batch_id * (HV * V_DIM * K_DIM) + v_head_id * (V_DIM * K_DIM);
    const int v_head_base = batch_id * (HV * V_DIM) + v_head_id * V_DIM;
    const int v_base = i_vb * BV_TOTAL + warp_id * BV;

    // ===== Load state via float4 + decay =====
    float s_regs[BV][KPT];

    #pragma unroll
    for (int r = 0; r < BV; r++) {
        const int v_idx = v_base + r;
        if (v_idx < V_DIM) {
            const float4* s_row_f4 = reinterpret_cast<const float4*>(
                S + s_head_offset + v_idx * K_DIM);
            float4 tmp = s_row_f4[lane_id];
            s_regs[r][0] = tmp.x * decay;
            s_regs[r][1] = tmp.y * decay;
            s_regs[r][2] = tmp.z * decay;
            s_regs[r][3] = tmp.w * decay;
        } else {
            #pragma unroll
            for (int i = 0; i < KPT; i++) s_regs[r][i] = 0.0f;
        }
    }

    // ===== Load v via uint2 (64-bit) =====
    float v_vals[BV];
    {
        const int vb_idx = v_head_base + v_base;
        if (v_base + BV <= V_DIM) {
            const uint2* v_u2 = reinterpret_cast<const uint2*>(V_in + vb_idx);
            uint2 v_packed = v_u2[0];
            const __nv_bfloat162* v_bf2 = reinterpret_cast<const __nv_bfloat162*>(&v_packed);
            v_vals[0] = __bfloat162float(v_bf2[0].x);
            v_vals[1] = __bfloat162float(v_bf2[0].y);
            v_vals[2] = __bfloat162float(v_bf2[1].x);
            v_vals[3] = __bfloat162float(v_bf2[1].y);
        } else {
            #pragma unroll
            for (int r = 0; r < BV; r++) {
                const int v_idx = v_base + r;
                v_vals[r] = (v_idx < V_DIM)
                    ? __bfloat162float(V_in[vb_idx + r]) : 0.0f;
            }
        }
    }

    // ===== Delta rule =====
    #pragma unroll
    for (int r = 0; r < BV; r++) {
        float partial_stk = 0.0f;
        #pragma unroll
        for (int j = 0; j < KPT; j++) {
            partial_stk += s_regs[r][j] * r_k[j];
        }
        float stk = warp_reduce_sum(partial_stk);
        float delta = beta * (v_vals[r] - stk);
        #pragma unroll
        for (int j = 0; j < KPT; j++) {
            s_regs[r][j] += delta * r_k[j];
        }
    }

    // ===== Store state via float4 + compute output =====
    #pragma unroll
    for (int r = 0; r < BV; r++) {
        const int v_idx = v_base + r;
        if (v_idx < V_DIM) {
            float4* dst_f4 = reinterpret_cast<float4*>(
                New_S + s_head_offset + v_idx * K_DIM);
            float4 out_f4;
            out_f4.x = s_regs[r][0];
            out_f4.y = s_regs[r][1];
            out_f4.z = s_regs[r][2];
            out_f4.w = s_regs[r][3];
            dst_f4[lane_id] = out_f4;

            float partial_o = 0.0f;
            #pragma unroll
            for (int j = 0; j < KPT; j++) {
                partial_o += s_regs[r][j] * r_q[j];
            }
            float o_val = warp_reduce_sum(partial_o);
            if (lane_id == 0) {
                O[v_head_base + v_idx] = __float2bfloat16(o_val);
            }
        }
    }
}


// =============================================================================
// TVM FFI Host Function — DPS (Destination Passing Style)
// =============================================================================
void DeltaNetRecurrentForward(
    tvm::ffi::TensorView q,         // [B, T, H, K]   bf16
    tvm::ffi::TensorView k,         // [B, T, H, K]   bf16
    tvm::ffi::TensorView v,         // [B, T, HV, V]  bf16
    tvm::ffi::TensorView state,     // [B, HV, V, K]  f32
    tvm::ffi::TensorView A_log,     // [HV]           f32
    tvm::ffi::TensorView a,         // [B, T, HV]     bf16
    tvm::ffi::TensorView dt_bias,   // [HV]           f32
    tvm::ffi::TensorView b,         // [B, T, HV]     bf16
    double scale,                    // float scalar (TVM FFI passes floats as double)
    tvm::ffi::TensorView output,    // [B, T, HV, V]  bf16 (DPS pre-allocated)
    tvm::ffi::TensorView new_state  // [B, HV, V, K]  f32  (DPS pre-allocated)
) {
    const int B = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(2));
    const int HV = static_cast<int>(v.size(2));
    const int V_val = static_cast<int>(v.size(3));

    const int NV_BLOCKS = (V_val + BV_TOTAL - 1) / BV_TOTAL;

    // Get CUDA stream from TVM FFI environment
    DLDevice dev = q.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    dim3 grid(NV_BLOCKS, B * HV);
    dim3 block(BLOCK_SIZE);

    deltanet_recurrent_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<const float*>(A_log.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
        reinterpret_cast<const float*>(dt_bias.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
        reinterpret_cast<const float*>(state.data_ptr()),
        reinterpret_cast<float*>(new_state.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<float>(scale),
        H,
        HV
    );
}

// Export the function as "kernel" for TVM FFI
TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, DeltaNetRecurrentForward);
