#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define NUM_WORKERS (1) 
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

constexpr int ATTN_B = 1;
constexpr int ATTN_H = 1;
constexpr int N = 2048;
constexpr int head_dim =  64;                   
constexpr float rope_embd_fraction = 1.0f;

// Make these constexpr so they can be used as template parameters
constexpr int rope_dim = static_cast<int>(rope_embd_fraction * head_dim);
constexpr int half_rope_dim = ( rope_dim / 2 );
constexpr int excess_dim = head_dim - rope_dim; 

constexpr int N_CHUNK  = 32;
constexpr int seq_tiles = N_CHUNK;
constexpr int rope_tiles = rope_dim;
constexpr int half_rope_tiles = half_rope_dim;
constexpr int excess_rope_tiles = excess_dim;

using namespace kittens;

#define tile_1xFULL_ROPE_D st<bf16, seq_tiles, rope_tiles>
#define tile_1xHALF_ROPE_D st<bf16, seq_tiles, half_rope_tiles>
#define tile_1xEXCESS_ROPE_D st<bf16, seq_tiles, excess_rope_tiles>

#define reg_tile_1xFULL_ROPE_D rt_bf<seq_tiles, rope_tiles>
#define reg_tile_1xHALF_ROPE_D rt_bf<seq_tiles, half_rope_tiles>
#define reg_tile_1xEXCESS_ROPE_D rt_bf<seq_tiles, excess_rope_tiles>


template<int _d_model> struct rotary_globals {
    static constexpr int d_model = _d_model;

    // global descriptors
    using x_gl            = gl<bf16, -1, -1, -1, -1>;
    using o_gl            = gl<bf16, -1, -1, -1, -1>;
    using sin_gl          = gl<bf16, -1, -1, -1, -1>;
    using cos_gl          = gl<bf16, -1, -1, -1, -1>;

    // global pointers
    x_gl x;
    o_gl o;
    sin_gl sin;
    cos_gl cos;

    dim3 grid() { return dim3(ATTN_B, ATTN_H); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY / 2; }
};



template<int D> __launch_bounds__(NUM_THREADS, 2)
__global__ void _fused_rotary( const rotary_globals<D> g) {
    auto warpid = kittens::warpid();
    auto lane = kittens::laneid();

    const int b = blockIdx.x;
    const int h = blockIdx.y;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    tile_1xFULL_ROPE_D (&x_s)       = al.allocate<tile_1xFULL_ROPE_D>();    
    tile_1xHALF_ROPE_D (&cos_s)     = al.allocate<tile_1xHALF_ROPE_D>(); 
    tile_1xHALF_ROPE_D (&sin_s)     = al.allocate<tile_1xHALF_ROPE_D>(); 

    int tic = 0, toc = 1;    
    int n_blocks = N / (NUM_WORKERS*N_CHUNK);
    for (int block = 0; block < n_blocks; block ++, tic ^=1, toc ^=1) {

        // smem loads
        load(x_s,  g.x, {b, h, block, 0});
        load(cos_s, g.cos, {0, 0, block, 0});
        load(sin_s, g.sin, {0, 0, block, 0});

        // register loads
        reg_tile_1xHALF_ROPE_D cos, sin, x1, x2, temp1, temp2;
        reg_tile_1xFULL_ROPE_D x;
        load(x, x_s);
        load(cos, cos_s);
        load(sin, sin_s);

        const int x_width = x.width;
        const int x1_width = x1.width;
        // for(int i = 0; i < head_dim/32; i++) {
        //     #pragma unroll
        //     for(int j = 0; j < 4; j++) {
        //         x1.tiles[0][i].data[j] = x.tiles[0][i].data[j];
        //         x2.tiles[0][i].data[j] = x.tiles[0][i+head_dim/32].data[j];
        //     }
        // }
        
        // a = torch.cat((x1, x2), dim=-1) * repeat(cos_in, "n d -> 1 n (2 d)" ) 
        mul(temp1, x1, cos);
        mul(temp2, x2, cos);

        // b = torch.cat((-x2, x1), dim=-1)  * repeat(sin_in, "n d -> 1 n (2 d)" )
        mul(x2, x2, -1.f);
        mul(x2, x2, sin);
        mul(x1, x1, sin);

        // sum ( a + b )
        add(temp1, temp1, x2);
        add(temp2, temp2, x1);

        // assemble the result
        zero(x);
        // for(int i = 0; i < head_dim/32; i++) {
        //     #pragma unroll
        //     for(int j = 0; j < 4; j++) {
        //         x.tiles[0][i].data[j]            = temp1.tiles[0][i].data[j];
        //         x.tiles[0][i+head_dim/32].data[j] = temp2.tiles[0][i].data[j];
        //     }
        // }
        
        // store out
        store(g.o, x, {b, h, block, 0});
    }
}

template<int D>
void dispatch_rotary(rotary_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)_fused_rotary<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    _fused_rotary<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_rotary<head_dim>>(m, "dispatch_rotary", 
        &rotary_globals<head_dim>::x, 
        &rotary_globals<head_dim>::o, 
        &rotary_globals<head_dim>::sin, 
        &rotary_globals<head_dim>::cos
    );
}

