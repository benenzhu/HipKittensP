// 



























// #ifndef PYTHON_CALL
// DeepSeek-V2/V3 MLA 参数
constexpr int BATCH = 4;
constexpr int H_Q = 128;          // query heads
constexpr int H_KV = 1;           // kv heads (MLA absorbed)
constexpr int DV__512 = 512;           // value dim (kv_lora_rank)
constexpr int DPE = 64;           // rope dim
constexpr int PAGE_BLOCK_SIZE = 64;  // paged attention block size
constexpr int BLOCK_H__64 = 64;       // heads per thread block
constexpr int BLOCK_N__64 = 64;       // KV tokens per iteration
constexpr int SEQ_LEN = 4096;
// constexpr int MAX_SEQLEN = 8192;
// #endif

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

// Kernel 配置
constexpr int NUM_WARPS = 8;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;  // 4 * 64 = 512

// Shared memory tile types (with swizzle for bank conflict reduction)
// Q tiles: [BLOCK_H, DV] = [64, 512] 和 [BLOCK_H, DPE] = [64, 64]
// using ST_Q = st_bf<BLOCK_H, DV, st_16x16_s>;           // Q nope
// using ST_Q_pe = st_bf<BLOCK_H, DPE, st_16x16_s>;       // Q rope

// KV tiles: [BLOCK_N, DV] = [64, 512] 和 [BLOCK_N, DPE] = [64, 64]
// using ST_KV = st_bf<BLOCK_N, DV, st_16x16_s>;          // KV (V is first DV dims)
// using ST_K_pe = st_bf<BLOCK_N, DPE, st_16x16_s>;       // K rope

// // Score tile: [BLOCK_H, BLOCK_N] = [64, 64]
// using ST_S = st_bf<BLOCK_H, BLOCK_N, st_16x16_s>;

// // Per-warp register tiles
constexpr int WARP_H = BLOCK_H__64 / NUM_WARPS;  // 64 / 4 = 16 rows per warp

// // Register tiles for accumulation (float32)
// using RT_S = rt_fl<WARP_H, BLOCK_N, row_l>;      // [16, 64] attention scores
// using RT_O = rt_fl<WARP_H, DV, row_l>;           // [16, 512] output accumulator

// // Register tiles for online softmax (per-row scalars)
// using RT_rowvec = rt_fl<WARP_H, 1, row_l>;       // [16, 1] for max/sum/scale

// // Global layout types
// using GL_Q = gl<bf16, 1, 1, -1, DV>;              // [batch, h_q, dv]
// using GL_Q_pe = gl<bf16, 1, 1, -1, DPE>;          // [batch, h_q, dpe]
// using GL_KV = gl<bf16, 1, 1, -1, DV>;             // [total_tokens, h_kv, dv]
// using GL_K_pe = gl<bf16, 1, 1, -1, DPE>;          // [total_tokens, h_kv, dpe]
// using GL_O = gl<bf16, 1, 1, -1, DV>;              // [batch, h_q, dv]

__device__ bool thread0() {
    return threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0;
}

/**
 * FlashMLA Paged Decoding Kernel
 * 
 * Computes: Output = softmax(Q @ K^T / sqrt(d)) @ V
 * Where Q = [Q_nope, Q_rope], K = [K_nope, K_rope], V = KV[:, :, :DV]
 * 
 * Key optimizations:
 * 1. Online softmax (Flash Attention style) - O(1) extra memory
 * 2. Pipelined KV loading with double buffering
 * 3. Separate Q_nope/Q_rope and K_nope/K_rope computation
 * 4. Paged attention via block_table indirect addressing
 * 5. Reverse iteration for numerical stability
 */



 using GL_Q = gl<bf16, 1, BATCH, H_Q, DV__512>;
 using GL_QPE = gl<bf16, 1, BATCH, H_Q, DPE>;
 using GL_KV = gl<bf16, 1, BATCH, SEQ_LEN, DV__512>;
 using GL_KVPE = gl<bf16, 1, BATCH, SEQ_LEN, DPE>;
//  using GL_TABLE = gl<int, 1, 1, BATCH, SEQ_LEN>;
 using GL_O = gl<bf16, 1, BATCH, H_Q, DV__512>;
 using G = kittens::group<NUM_WARPS>;
 

#define BARRIER { \
    asm volatile("s_waitcnt vmcnt(0)"); \
    asm volatile("s_waitcnt lgkmcnt(0)"); \
    __syncthreads(); \
}
 #define D(x) do { if(thread0())printf("%d,  " #x ": %lf\n",  __LINE__, static_cast<float>(x)); } while (0)
 #define Dk(x) do { if(thread0())printf("%d, K:%d " #x ": %lf\n",  __LINE__, k, static_cast<float>(x)); } while (0)
//  #define D(x) 
//  #define Dk(x) 
 #define Dk2(x) do { if(thread0())printf("%d, K:%d " #x ": %lf\n",  __LINE__, k, static_cast<float>(x)); } while (0)
 

template<typename D>
__device__ bf16 sum_tile(D A2_tile){ 
    bf16 sum_val_A = 0;
    #pragma unroll
    for(int i = 0; i < A2_tile.height; i++){
        #pragma unroll
        for(int j = 0; j < A2_tile.width; j++){
            // for(int k = 0)
            #pragma unroll
            for(int k = 0; k < A2_tile.packed_per_base_tile; k++){
                sum_val_A += A2_tile.tiles[i][j].data[k].x;
                // sum_val_A += A2_tile.tiles[i][j].data[k].y;
            }
        }
    }

    return sum_val_A;

}


template<typename D>
__device__ bf16 sum_row(D o_scores_sum){ 
    bf16 sum_o_scores = 0;
    for(int i = 0; i < o_scores_sum.outer_dim; i++){
        for(int j = 0; j < o_scores_sum.inner_dim; j++){
            sum_o_scores += o_scores_sum.data[i][j].x;
            sum_o_scores += o_scores_sum.data[i][j].y;
        }
    }
    return sum_o_scores;
}



__global__ 
__launch_bounds__(NUM_THREADS, 2)
void flashmla_paged_decoding(
    bf16* __restrict__ q_ptr,              // [batch, h_q, dv]
    bf16* __restrict__ qpe_ptr, 
    bf16* __restrict__ kv_ptr,
    bf16* __restrict__ kvpe_ptr,
    bf16* __restrict__ output_ptr         // [batch, h_q, dv]
    // int* __restrict__ block_table,     // [batch, max_num_blocks]
    // bf16* __restrict__ blocked_kv,             // [total_tokens, h_kv, dv]
    // int* __restrict__ cache_seqlens,   // [batch]
    // int max_num_blocks                 // max blocks per batch
) {
    GL_Q Q = GL_Q(q_ptr);
    GL_QPE QPE = GL_QPE(qpe_ptr);
    GL_KV KV = GL_KV(kv_ptr);
    GL_KVPE KVPE = GL_KVPE(kvpe_ptr);
    GL_O output = GL_O(output_ptr);

    // Block indices
    const int batch_idx = blockIdx.x;
    const int head_group = blockIdx.y;  // processes BLOCK_H heads at a time
    __builtin_assume(head_group <= 2);
    const int warp_id = kittens::warpid();
    const int lane_id = kittens::laneid();
    
    // Shared memory allocation
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    // Double-buffered KV tiles
    using ST_Q = st_bf<BLOCK_H__64, DV__512, st_16x32_s>; // 64 * 512
    ST_Q& shared_Q = al.allocate<ST_Q>();

    using ST_S = st_bf<BLOCK_H__64, BLOCK_N__64, st_16x16_s>;
    ST_S& S_shared = al.allocate<ST_S>();

 
    using ST_KV = st_bf<BLOCK_N__64, DV__512, st_16x16_s>;          // KV (V is first DV dims)
    ST_KV (&shared_KV) = al.allocate<ST_KV>();
    
    using ST_ACC_S = st_bf<BLOCK_H__64, BLOCK_N__64, st_16x16_s>;
    ST_ACC_S& shared_s = al.allocate<ST_ACC_S>();
    
    struct ___{
       float data[64];
    };
    
    
    ___& shared_scale = al.allocate<___>();
    

    constexpr auto total_shared_sizes__147456 = sizeof(shared_Q) + sizeof(S_shared) + sizeof(shared_KV) + sizeof(shared_s);
                                
    static_assert(total_shared_sizes__147456 <= 160000, "Shared memory size exceeds 160KB");
    


    // 64 * 64 / 8 = 32 * 16
    rt_fl<16, 32, col_l, rt_16x16_s> acc_s;
    rt_fl<8, 64, col_l, rt_8x64_s> acc_s_trans;
    
    //  8 * 32
    

    // g2r for Q & Q_pe :::[64, 576]
    // G::load(Q_shared, Q, {})
    G::load(shared_Q, Q, {0, batch_idx, head_group, 0});
    BARRIER;
    // for k in seq_len // BN
    int num_kv_blocks = (SEQ_LEN + BLOCK_N__64 - 1) / BLOCK_N__64;
    rt_bf<16, 32, row_l, rt_16x32_s> A_tile;
    rt_bf<32, 32, row_l, rt_16x32_s> B_tile;
    rt_bf<64, 64, row_l, rt_16x32_s> A2_tile;
    rt_bf<64, 64, row_l, rt_16x32_s> B2_tile;
    rt_fl<64, 64, col_l, rt_16x16_s> acc_o; // 2row, 4col.
    typename decltype(acc_s_trans)::row_vec max_vec, max_vec_prev, scores_sum, log_sum, scale_vec;
    typename decltype(acc_o)::col_vec o_scale_vec, o_scores_sum;
    zero(acc_s);
    ones(scale_vec);
    constexpr auto now___ = sizeof(max_vec);
    constexpr int regnum = sizeof(acc_o) / sizeof(float); 

    const int warp_row__0_3 = warp_id / 2;
    const int warp_col__0_1 = warp_id % 2;
    
    neg_infty(max_vec);
    zero(acc_o);
    zero(log_sum);
    zero(o_scores_sum);
    
    
    
    

    // for (int k = 0; k < num_kv_blocks - 2; k++){
    // for (int k = 0; k < num_kv_blocks - 60; k++){
    for (int k = 0; k < 2; k++){
        //  KV_shared = T.copy(blocked_kv)::: [64, 576]
        // if(thread0())printf("batch_idx:%d, pos: %d, kv_ptr %p DV: %d\n", batch_idx, k * BLOCK_N, kv_ptr, DV);
        G::load(shared_KV, KV, {0, batch_idx, k, 0});
        
        constexpr float now = 16777216.0 / (4 * 4096 * 512) / 2;
        constexpr float now2 = 212336640.0 / (4 * 4096 * 512) / 2;
        BARRIER;
        

        #ifdef hip_rtc
        if(thread0()){
            float shared_Q_sum = 0;
            for(int i = 0; i < 32768; i++){
                shared_Q_sum += float(shared_Q.data[i]);
            }
            Dk(shared_Q_sum);
            float shared_K_sum = 0;
            for(int i = 0; i < 32768; i++){
                shared_K_sum += float(shared_KV.data[i]);
            }
            Dk(shared_K_sum);
        }
        __syncthreads();
        #endif
        // 1. acc_s = T.gemm(Q @ KV_shared) ::: [64, 64]
        zero(acc_s);
        for(int step = 0; step < DV__512/32; step += 1){
            // load A (16 * 32)
            // load B (32 * 32)
            // C: (16 * 32)
            // auto ret = subtile_inplace<16, 32>(shared_Q, {warp_row, step});
            load(A_tile, subtile_inplace<16, 32>(shared_Q, {warp_row__0_3, step}));
            load(B_tile, subtile_inplace<32, 32>(shared_KV, {warp_col__0_1, step}));
            BARRIER;
            mma_ABt(acc_s, A_tile, B_tile, acc_s);
            
        }
        #ifdef hip_rtc2
        if(k == 2){
            D(acc_s.tiles[0][0].data[0].x);
            D(acc_s.tiles[0][0].data[0].y);
            D(acc_s.tiles[0][0].data[1].x);
            D(acc_s.tiles[0][0].data[1].y);
            D(float(A_tile.tiles[0][0].data[0].x));
            D(float(shared_Q.data[0]));
        }
        #endif
        
        // acc_s layout: [16,32] col
        
        /*
            acc_s: [16, 32]   [shared_s] [64, 64];
            
        */
        auto r2s = [&](){
            for(int i = 0; i < acc_s.width; i++){
                shared_s.data[(warp_row__0_3 * 16 + lane_id % 4 * 4) * 64 +     i * 16 + warp_col__0_1 * 32 + lane_id / 4] = acc_s.tiles[0][i].data[0].x;
                shared_s.data[(warp_row__0_3 * 16 + lane_id % 4 * 4 + 1) * 64 + i * 16 + warp_col__0_1 * 32 + lane_id / 4] = acc_s.tiles[0][i].data[0].y;
                shared_s.data[(warp_row__0_3 * 16 + lane_id % 4 * 4 + 2) * 64 + i * 16 + warp_col__0_1 * 32 + lane_id / 4] = acc_s.tiles[0][i].data[1].x;
                shared_s.data[(warp_row__0_3 * 16 + lane_id % 4 * 4 + 3) * 64 + i * 16 + warp_col__0_1 * 32 + lane_id / 4] = acc_s.tiles[0][i].data[1].y;
            }
            BARRIER;
        };
        r2s();
        __syncthreads();
        
        auto s2r = [&]() {
            for(int i = 0; i < 4; i++){
                acc_s_trans.tiles[0][0].data[i].x = shared_s.data[(warp_id * 8 + lane_id % 8) * 64 + (lane_id / 8) * 8 + i * 2];
                acc_s_trans.tiles[0][0].data[i].y = shared_s.data[(warp_id * 8 + lane_id % 8) * 64 + (lane_id / 8) * 8 + i * 2 + 1];
                // if(k == 0)
                    // Dk(acc_s_trans.tiles[0][0].data[0].x);
                    // Dk(acc_s_trans.tiles[0][0].data[0].y);
                // }
            }
            Dk(sum_tile(acc_s_trans));
            BARRIER;
        };
        s2r();
        


        
        

        // need to trans to [8, 64] col/row

        // store(subtile_inplace<16, 32>(shared_s, {warp_row, warp_col}), acc_s);

        //TODO: add a gemm for QPE @ KPE here..
        // 更新最大值
        // 2.1 max_vec_prev = max_vec
        copy(max_vec_prev, max_vec);
        // 2.2 max_vec = T.row_max(acc_s)
        // 3. max_vec = max(max_vec, max_vec_prev)
        col_max(max_vec, acc_s_trans, max_vec_prev);
        if(k == 0){
            Dk2(acc_s_trans.tiles[0][0].data[0].x);
            Dk2(max_vec.data[0][0]);
        }
        
        // 计算缩放因子，作用于归一化分母(sum), 输出(O) acc_s直接减去最大值，然后exp2, 乘上v加起来就行.
        // 4. scale_vec = max_vec_prev - max_vec
        sub(scale_vec, max_vec_prev, max_vec);       // scale_vec = old_max - new_max
        // 5. scale_vec = T.exp2(scores_vec)
        exp2(scale_vec, scale_vec);      

        // 6. acc_s -= max_vec 
        // TODO(zty)::::
        sub_col(acc_s_trans, acc_s_trans, max_vec);
        
        // 7. acc_s = T.exp2(acc_s)
        exp2(acc_s_trans, acc_s_trans);

        // 8. acc_s_shared = acc_s 
        // TODO(zty):
        // store(shared_s, acc_s);
        
        // 8.1 scores_sum = T.row_sum(acc_s)
        // TODO:
        col_sum(scores_sum, acc_s_trans, scores_sum);

        // 8.2 logsum *=scale_vec 
        mul(log_sum, log_sum, scale_vec);
        // 8.3 logsum += scores_sum
        add(log_sum, log_sum, scores_sum);

        // 9 acc_o *= scores_vec
        if(lane_id / 8 == 0) shared_scale.data[warp_id * 8 + lane_id / 8] = scale_vec.data[0][0];
        __syncthreads();
        
        
        for(int i = 0; i < 4; i++){
            o_scale_vec.data[i][0].x = shared_scale.data[i * 16 + lane_id % 4 * 4];
            o_scale_vec.data[i][0].y = shared_scale.data[i * 16 + lane_id % 4 * 4 + 1];
            o_scale_vec.data[i][1].x = shared_scale.data[i * 16 + lane_id % 4 * 4 + 2];
            o_scale_vec.data[i][1].y = shared_scale.data[i * 16 + lane_id % 4 * 4 + 3];
        }

        auto r2s_trans = [&]() {
            for(int i = 0; i < 4; i++){
                shared_s.data[(warp_id * 8 + lane_id % 8) * 64 + (lane_id / 8) * 8 + i * 2] = acc_s_trans.tiles[0][0].data[i].x;
                shared_s.data[(warp_id * 8 + lane_id % 8) * 64 + (lane_id / 8) * 8 + i * 2 + 1] = acc_s_trans.tiles[0][0].data[i].y;
                // if(k == 0){
                //     D(acc_s_trans.tiles[0][0].data[0].x);
                //     D(acc_s_trans.tiles[0][0].data[0].y);
                // }
            }
            BARRIER;
        };

        r2s_trans();        
        
        BARRIER;
        

        mul_row(acc_o, acc_o, o_scale_vec);

        if(k < 1000){
            Dk2(acc_o.tiles[0][0].data[0].x);
        }

        if(thread0()){
            float shared_s_sum = 0;
            for(int i = 0; i < 4096; i++){
                shared_s_sum += float(shared_s.data[i]);
            }
            Dk2(shared_s_sum);
            float shared_K_sum = 0;
            for(int i = 0; i < 32768; i++){
                shared_K_sum += float(shared_KV.data[i]);
            }
            Dk2(shared_K_sum);
        }

        {
            load(A2_tile, subtile_inplace<64,64>(shared_s, {0,0}));
            load(B2_tile, subtile_inplace<64,64>(shared_KV, {0, kittens::warpid()}));
            A2_tile.tiles[0][0].data[0].x += 0.0;
            // Dk(A2_tile.tiles[0][0].data[0].y);
            //
            BARRIER;
            bf16 sum_val_A = sum_tile(A2_tile);
            // bf16 sum_val_A = 0;
            bf16 sum_val_B = sum_tile(B2_tile);
            // for(int i = 0; i < 4; i++){
            //     for(int j = 0; j < 2; j++){
            //         // for(int k = 0)
            //         for(int k = 0; k < A2_tile.packed_per_base_tile; k++){
            //             sum_val_A += A2_tile.tiles[i][j].data[k].x;
            //             sum_val_A += A2_tile.tiles[i][j].data[k].y;
            //         }
            //     }
            // }
            Dk2(sum_val_A);
            Dk2(sum_val_B);
            mma_ABt(acc_o, A2_tile, B2_tile, acc_o);
            Dk2(sum_row(o_scores_sum));
        }
        if(k < 1000){
            Dk2(acc_o.tiles[0][0].data[0].x);
        }

    }
    

    D(sum_tile(acc_o));
    D(sum_row(o_scores_sum));
    div_row(acc_o, acc_o, o_scores_sum);
    D(sum_tile(acc_o));
    rt_fl<64, 64, row_l, rt_16x16_s> o_reg_transposed; // 2row, 4col. 
    transpose(o_reg_transposed, acc_o);
    D(sum_tile(o_reg_transposed));
    if(thread0()){
        printf("output1: %lf\n", acc_o.tiles[0][0].data[0].x);
    }
    
    // TODO(zty):::::: 
    zhuzhustore::store(output, o_reg_transposed, {0, batch_idx, head_group, 0});
}

// Host-side launcher (for non-RTC usage)
#ifndef hip_rtc
struct flashmla_globals {
    GL_Q q;
    GL_Q_pe q_pe;
    GL_KV kv;
    GL_K_pe k_pe;
    int* block_table;
    int* cache_seqlens;
    GL_O output;
    int batch;
    int max_num_blocks;
    
    dim3 grid() { return dim3(batch, H_Q / BLOCK_H, 1); }
    dim3 block() { return dim3(NUM_THREADS); }
    
    size_t dynamic_shared_memory() {
        // Q: 64*512*2 + 64*64*2 = 65536 + 8192 = 73728
        // KV double buffer: 2*(64*512*2 + 64*64*2) = 147456
        // S: 64*64*2 = 8192
        // Total: ~230KB, but we can optimize
        return 160000;  // 160KB safe estimate
    }
};
#endif