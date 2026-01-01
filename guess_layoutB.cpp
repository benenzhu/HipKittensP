#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>


// Input vector: 4 VGPRs = 128 bits. Holds 8 x bf16.
typedef float v4f32 __attribute__((vector_size(16)));
typedef unsigned short v8bf16 __attribute__((vector_size(16))); // Representation

// Output vector: 4 VGPRs = 128 bits. Holds 4 x f32.
typedef float v4f32_out __attribute__((vector_size(16)));

// -----------------------------------------------------------------------------
// Kernel
// -----------------------------------------------------------------------------

// Wrapper for the intrinsic to handle potential compiler differences
__device__ v4f32_out mfma_inst(v8bf16 a, v8bf16 b, v4f32_out c) {
    // Note: If your compiler does not yet support this builtin, you may need 
    // to use inline assembly or a newer ROCm version.
    // Assuming the user provided name is correct for the target compiler.
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
}
__global__ void guess_fp16(float* d_C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    

    v8bf16 a[8];
    v8bf16 b[8];
    if(threadIdx.x == 0){
        for(int i = 0; i < 8; i++){
            ((__hip_bfloat16*)a)[i] = i + 1;
        }
    }
    for(int i = 0; i < 8; i++){
        ((__hip_bfloat16*)b)[i] = 1;
    }

    
    v4f32_out c;
    for(int i = 0; i < 4; i++){
        ((float*)&c)[i] = 0;
    }
    mfma_inst(reinterpret_cast<v8bf16>(*a), reinterpret_cast<v8bf16>(*b), c);
    __syncthreads();
    if(threadIdx.x == 0){
        for(int i = 0; i < 8; i++){
            printf("%.2lf ", float(((__hip_bfloat16*)a)[i]));
        }
        printf("\n");
        for(int i = 0; i < 8; i++){
            printf("%.2lf ", float(((__hip_bfloat16*)b)[i]));
        }
        printf("\n");
        for(int i = 0; i < 8; i++){
            printf("%.2lf ", ((float*)&c)[i]);
        }
    }
    for(int i = 0; i < 4; i++){
        d_C[threadIdx.x * 4 + i] = ((float*)&c)[i];
    }
    
    // if(threadIdx.x == 0){
    //     d_C[1] = 1;
    // }

}


int main() {

    float* d_C;
    // 16, 16, 32
    auto ret = hipMalloc(&d_C, 256 * sizeof(float));
    
    guess_fp16<<<1, 64>>>(d_C);
    
    float h_C[256];
    ret = hipMemcpy(h_C, d_C, 256 * sizeof(float), hipMemcpyDeviceToHost);
    
    for(int i = 0; i < 16; i++){
        for(int j = 0; j < 16; j++){
            printf("%.2lf ", h_C[i * 16 + j]);
        }
        puts("");
    }

    
}