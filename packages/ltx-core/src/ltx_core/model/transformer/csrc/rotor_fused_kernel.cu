#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_GROUPS 1408
#define MAX_LEVELS 256

template <typename T>
__device__ float convert_to_float(T value) { return static_cast<float>(value); }
template <> __device__ float convert_to_float<c10::Half>(c10::Half value) { return __half2float(value); }
template <> __device__ float convert_to_float<at::BFloat16>(at::BFloat16 value) { return static_cast<float>(value); }

template <typename T>
__device__ T convert_from_float(float value) { return static_cast<T>(value); }
template <> __device__ c10::Half convert_from_float<c10::Half>(float value) { return __float2half(value); }
template <> __device__ at::BFloat16 convert_from_float<at::BFloat16>(float value) { return static_cast<at::BFloat16>(value); }

__device__ void gp_rotor_mv(float s, float p12, float p13, float p23,
                            const float x[8], float r[8]) {
    r[0] = s * x[0] - p12 * x[4] - p13 * x[5] - p23 * x[6];
    r[1] = s * x[1] + p12 * x[2] + p13 * x[3] + p23 * x[7];
    r[2] = s * x[2] - p12 * x[1] + p23 * x[3] - p13 * x[7];
    r[3] = s * x[3] - p13 * x[1] - p23 * x[2] + p12 * x[7];
    r[4] = s * x[4] + p12 * x[0];
    r[5] = s * x[5] + p13 * x[0];
    r[6] = s * x[6] + p23 * x[0];
    r[7] = s * x[7] - p23 * x[1] + p13 * x[2] - p12 * x[3];
}

__device__ float quantize(float val, const float* centroids, int levels) {
    float best = centroids[0];
    float min_d = fabsf(val - best);
    #pragma unroll
    for (int i = 1; i < MAX_LEVELS; ++i) {
        if (i >= levels) break;
        float d = fabsf(val - centroids[i]);
        if (d < min_d) { min_d = d; best = centroids[i]; }
    }
    return best;
}

template <typename T>
__global__ void rotor_full_fused_kernel(
    const T* __restrict__ input,
    const float* __restrict__ rotors,
    const float* __restrict__ c_scalar,   int n_scalar,
    const float* __restrict__ c_vector,   int n_vector,
    const float* __restrict__ c_bivector, int n_bivector,
    const float* __restrict__ c_trivector,int n_trivector,
    T* __restrict__ output,
    int batch_size, int emb_dim, int n_groups)
{
    __shared__ float sh_rotors[MAX_GROUPS * 4];
    __shared__ float sh_c_scalar[MAX_LEVELS];
    __shared__ float sh_c_vector[MAX_LEVELS];
    __shared__ float sh_c_bivector[MAX_LEVELS];
    __shared__ float sh_c_trivector[MAX_LEVELS];

    int tid = threadIdx.x;

    // Cooperative load rotors (float4 for efficiency)
    for (int i = tid; i < n_groups; i += blockDim.x) {
        int base = i * 4;
        sh_rotors[base + 0] = rotors[base + 0];
        sh_rotors[base + 1] = rotors[base + 1];
        sh_rotors[base + 2] = rotors[base + 2];
        sh_rotors[base + 3] = rotors[base + 3];
    }

    for (int i = tid; i < n_scalar; i += blockDim.x) sh_c_scalar[i] = c_scalar[i];
    for (int i = tid; i < n_vector; i += blockDim.x) sh_c_vector[i] = c_vector[i];
    for (int i = tid; i < n_bivector; i += blockDim.x) sh_c_bivector[i] = c_bivector[i];
    for (int i = tid; i < n_trivector; i += blockDim.x) sh_c_trivector[i] = c_trivector[i];

    __syncthreads();

    int b = blockIdx.x;
    if (b >= batch_size) return;

    const T* in_ptr = input + (size_t)b * emb_dim;
    T* out_ptr = output + (size_t)b * emb_dim;

    // Two arrays only — reuse mv for all stages to minimize registers
    for (int g = tid; g < n_groups; g += blockDim.x) {
        int d0 = g * 3;

        int ri = g * 4;
        float s   = sh_rotors[ri + 0];
        float p12 = sh_rotors[ri + 1];
        float p13 = sh_rotors[ri + 2];
        float p23 = sh_rotors[ri + 3];

        // Embed
        float mv[8] = {0.0f};
        float tmp[8];
        if (d0     < emb_dim) mv[1] = convert_to_float(in_ptr[d0]);
        if (d0 + 1 < emb_dim) mv[2] = convert_to_float(in_ptr[d0 + 1]);
        if (d0 + 2 < emb_dim) mv[3] = convert_to_float(in_ptr[d0 + 2]);

        // Forward sandwich → overwrite mv
        gp_rotor_mv(s, p12, p13, p23, mv, tmp);
        gp_rotor_mv(s, -p12, -p13, -p23, tmp, mv);

        // Quantize in-place
        mv[0] = quantize(mv[0], sh_c_scalar,   n_scalar);
        mv[1] = quantize(mv[1], sh_c_vector,   n_vector);
        mv[2] = quantize(mv[2], sh_c_vector,   n_vector);
        mv[3] = quantize(mv[3], sh_c_vector,   n_vector);
        mv[4] = quantize(mv[4], sh_c_bivector, n_bivector);
        mv[5] = quantize(mv[5], sh_c_bivector, n_bivector);
        mv[6] = quantize(mv[6], sh_c_bivector, n_bivector);
        mv[7] = quantize(mv[7], sh_c_trivector,n_trivector);

        // Inverse sandwich → overwrite mv
        gp_rotor_mv(s, -p12, -p13, -p23, mv, tmp);
        gp_rotor_mv(s, p12, p13, p23, tmp, mv);

        // Extract
        if (d0     < emb_dim) out_ptr[d0]     = convert_from_float<T>(mv[1]);
        if (d0 + 1 < emb_dim) out_ptr[d0 + 1] = convert_from_float<T>(mv[2]);
        if (d0 + 2 < emb_dim) out_ptr[d0 + 2] = convert_from_float<T>(mv[3]);
    }
}

template <typename T>
__global__ void rotor_sandwich_kernel(
    const T* __restrict__ input,
    const float* __restrict__ rotors,
    T* __restrict__ output,
    int batch_size, int emb_dim, int n_groups)
{
    int tid = threadIdx.x;
    int b = blockIdx.x;
    if (b >= batch_size) return;

    const T* in_ptr = input + (size_t)b * emb_dim;
    T* out_ptr = output + (size_t)b * n_groups * 8;

    for (int g = tid; g < n_groups; g += blockDim.x) {
        int ri = g * 4;
        float s   = rotors[ri]; float p12 = rotors[ri+1];
        float p13 = rotors[ri+2]; float p23 = rotors[ri+3];

        int d0 = g * 3;
        float mv[8] = {0.0f};
        float tmp[8];
        if (d0     < emb_dim) mv[1] = convert_to_float(in_ptr[d0]);
        if (d0 + 1 < emb_dim) mv[2] = convert_to_float(in_ptr[d0 + 1]);
        if (d0 + 2 < emb_dim) mv[3] = convert_to_float(in_ptr[d0 + 2]);

        gp_rotor_mv(s, p12, p13, p23, mv, tmp);
        gp_rotor_mv(s, -p12, -p13, -p23, tmp, mv);

        int base = g * 8;
        #pragma unroll
        for (int c = 0; c < 8; ++c)
            out_ptr[base + c] = convert_from_float<T>(mv[c]);
    }
}

template <typename T>
__global__ void rotor_inverse_sandwich_kernel(
    const T* __restrict__ input_mv,
    const float* __restrict__ rotors,
    T* __restrict__ output,
    int batch_size, int emb_dim, int n_groups)
{
    int tid = threadIdx.x;
    int b = blockIdx.x;
    if (b >= batch_size) return;

    const T* in_ptr = input_mv + (size_t)b * n_groups * 8;
    T* out_ptr = output + (size_t)b * emb_dim;

    for (int g = tid; g < n_groups; g += blockDim.x) {
        int ri = g * 4;
        float s   = rotors[ri]; float p12 = rotors[ri+1];
        float p13 = rotors[ri+2]; float p23 = rotors[ri+3];

        int base = g * 8;
        float mv[8], tmp[8];
        #pragma unroll
        for (int c = 0; c < 8; ++c)
            mv[c] = convert_to_float(in_ptr[base + c]);

        gp_rotor_mv(s, -p12, -p13, -p23, mv, tmp);
        gp_rotor_mv(s, p12, p13, p23, tmp, mv);

        int d0 = g * 3;
        if (d0     < emb_dim) out_ptr[d0]     = convert_from_float<T>(mv[1]);
        if (d0 + 1 < emb_dim) out_ptr[d0 + 1] = convert_from_float<T>(mv[2]);
        if (d0 + 2 < emb_dim) out_ptr[d0 + 2] = convert_from_float<T>(mv[3]);
    }
}

// ─── Dispatch ───

template <typename T>
torch::Tensor rotor_full_fused_impl(
    torch::Tensor input, torch::Tensor rotors,
    torch::Tensor c_scalar, int n_scalar,
    torch::Tensor c_vector, int n_vector,
    torch::Tensor c_bivector, int n_bivector,
    torch::Tensor c_trivector, int n_trivector)
{
    int batch_size = input.size(0);
    int emb_dim = input.size(1);
    int n_groups = (emb_dim + 2) / 3;
    auto output = torch::empty_like(input);

    // Use 256 threads to reduce register pressure per block
    int threads = min(256, max(n_groups, 32));
    rotor_full_fused_kernel<T><<<batch_size, threads>>>(
        input.data_ptr<T>(), rotors.data_ptr<float>(),
        c_scalar.data_ptr<float>(), n_scalar,
        c_vector.data_ptr<float>(), n_vector,
        c_bivector.data_ptr<float>(), n_bivector,
        c_trivector.data_ptr<float>(), n_trivector,
        output.data_ptr<T>(), batch_size, emb_dim, n_groups);
    return output;
}

template <typename T>
torch::Tensor rotor_sandwich_impl(torch::Tensor input, torch::Tensor rotors) {
    int batch_size = input.size(0);
    int emb_dim = input.size(1);
    int n_groups = (emb_dim + 2) / 3;
    auto output = torch::empty({batch_size, n_groups, 8}, input.options());
    int threads = min(256, max(n_groups, 32));
    rotor_sandwich_kernel<T><<<batch_size, threads>>>(
        input.data_ptr<T>(), rotors.data_ptr<float>(),
        output.data_ptr<T>(), batch_size, emb_dim, n_groups);
    return output;
}

template <typename T>
torch::Tensor rotor_inverse_impl(torch::Tensor input_mv, torch::Tensor rotors, int emb_dim) {
    int batch_size = input_mv.size(0);
    int n_groups = input_mv.size(1);
    auto output = torch::empty({batch_size, emb_dim}, input_mv.options());
    int threads = min(256, max(n_groups, 32));
    rotor_inverse_sandwich_kernel<T><<<batch_size, threads>>>(
        input_mv.data_ptr<T>(), rotors.data_ptr<float>(),
        output.data_ptr<T>(), batch_size, emb_dim, n_groups);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotor_full_fused_float", &rotor_full_fused_impl<float>);
    m.def("rotor_full_fused_half", &rotor_full_fused_impl<c10::Half>);
    m.def("rotor_full_fused_bf16", &rotor_full_fused_impl<at::BFloat16>);
    m.def("rotor_sandwich_float", &rotor_sandwich_impl<float>);
    m.def("rotor_sandwich_half", &rotor_sandwich_impl<c10::Half>);
    m.def("rotor_sandwich_bf16", &rotor_sandwich_impl<at::BFloat16>);
    m.def("rotor_inverse_float", &rotor_inverse_impl<float>);
    m.def("rotor_inverse_half", &rotor_inverse_impl<c10::Half>);
    m.def("rotor_inverse_bf16", &rotor_inverse_impl<at::BFloat16>);
}
