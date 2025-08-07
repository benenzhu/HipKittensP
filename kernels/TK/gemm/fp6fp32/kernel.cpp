#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define SIZE 64

using _gl_tile = gl<float, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    _gl_tile input;
    _gl_tile output;
    dim3 grid()  { return dim3(1); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; } 
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    // if (threadIdx.x  == 0 && blockIdx.x == 0) {
    //     printf("sizeof(fp6_e2m3): %zu\n", sizeof(fp6_e2m3));
    //     printf("sizeof(float): %zu\n", sizeof(float));
    //     printf("fp6 TILE_COL_DIM: %d\n", kittens::TILE_COL_DIM<fp6_e2m3>);
    //     printf("float TILE_COL_DIM: %d\n", kittens::TILE_COL_DIM<float>);
    //     // printf("fp6 packed_per_thread: %d\n", src.tiles[0][0].packed_per_thread);
    //     // printf("float packed_per_thread: %d\n", dst.tiles[0][0].packed_per_thread);
    //     // printf("src width: %d, dst width: %d\n", src.width, dst.width);
    // }


    rt_fl<SIZE, SIZE> tile_fl;
    rt_f6<SIZE, SIZE> tile_f6;
    one(tile_f6);
    copy(tile_fl, tile_f6);
    store(g.output, tile_fl, {0, 0, 0, 0});
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module"; 
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::input, &micro_globals::output);
}
