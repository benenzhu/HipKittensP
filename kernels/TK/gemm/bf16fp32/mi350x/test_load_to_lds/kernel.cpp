#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "../utils.cpp"
using namespace kittens;

constexpr int BLOCK_SIZE_ROWS = 64;
constexpr int BLOCK_SIZE_COLS = 64;  

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

#define USE_REF 0
#define USE_BASE 1

struct micro_globals {
    _gl_A in;
    _gl_C out, ref_out;
    dim3 grid()  { return dim3(1); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS> (&In) = al.allocate<st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS>>();
    st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS> (&Out) = al.allocate<st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS>>();

    st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS> (&In_ref) = al.allocate<st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS>>();
    st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS> (&Ref_Out) = al.allocate<st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS>>();

    rt_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS> in_reg, in_reg_ref;
    rt_fl<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS, ducks::rt_layout::col> out_reg, out_reg_ref;
    zero(out_reg);
    zero(out_reg_ref);

    // global to shared
    if (USE_BASE == 1)
        load_global_to_shared_direct<2, false, st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS>, _gl_A, coord<st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS>>, NUM_THREADS>(g.in, {0, 0, 0, 0}, In);
    if (USE_REF == 1)
        G::load(In_ref, g.in, {0, 0, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();

    // shared to registers
    if (USE_BASE == 1)
        // load_linear(in_reg, In);
        // load(in_reg, In);
        load_lds_reg(in_reg, In);
    if (USE_REF == 1)
        load(in_reg_ref, In_ref);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();

    // compute
    // __builtin_amdgcn_s_setprio(1);
    // if (USE_BASE == 1)
    //     mma_ABt(out_reg, in_reg, in_reg, out_reg);
    // if (USE_REF == 1)
    //     mma_ABt(out_reg_ref, in_reg_ref, in_reg_ref, out_reg_ref);
    // __builtin_amdgcn_s_barrier();
    // __builtin_amdgcn_sched_barrier(0);
    // __syncthreads();

    // register to global
    if (USE_BASE == 1)
        store(g.out, in_reg, {0, 0, 0, 0});
    if (USE_REF == 1)
        store(g.ref_out, in_reg_ref, {0, 0, 0, 0});
    __syncthreads();
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::in, &micro_globals::out, &micro_globals::ref_out);
}