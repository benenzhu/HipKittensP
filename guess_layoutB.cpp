#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>


typedef __bf16 v8bf16 __attribute__((vector_size(16))) ; // Representation
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
    

    v8bf16 a;
    v8bf16 b;
    if(threadIdx.x == 0){
        for(int i = 0; i < 8; i++){
            a[i] = i + 1;
        }
    }else{
        for(int i = 0; i < 8; i++){
            a[i] = 0;
        }
    }
    if (threadIdx.x == 0){
        for(int i = 0; i < 8; i++){
            b[i] = 1;
        }
    }else{
        for(int i = 0; i < 8; i++){
            b[i] = 0;
        }
    }

    
    v4f32_out c;
    for(int i = 0; i < 4; i++){
        c[i] = 0;
    }
    c = mfma_inst(a, b, c);
    __syncthreads();
    for(int i = 0; i < 4; i++){
        const int row = threadIdx.x / 16 * 4;
        const int col = threadIdx.x % 16;
        
        
        d_C[(row + i) * 16 + col] = c[i];
    }
    if(threadIdx.x == 0){
        for(int i = 0; i < 8; i++){
            printf("%.2lf ", float(a[i]));
        }
        printf("\n");
        for(int i = 0; i < 8; i++){
            printf("%.2lf ", float(b[i]));
        }
        printf("\n");
        for(int i = 0; i < 4; i++){
            printf("%.2lf ", c[i]);
        }
        printf("\n");
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