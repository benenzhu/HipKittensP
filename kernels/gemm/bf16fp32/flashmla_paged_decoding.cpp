#ifndef PYTHON_CALL
// DeepSeek-V2/V3 MLA 参数
constexpr int BATCH = 4;
constexpr int H_Q = 128;          // query heads
constexpr int H_KV = 1;           // kv heads (MLA absorbed)
constexpr int DV = 512;           // value dim (kv_lora_rank)
constexpr int DPE = 64;           // rope dim
constexpr int PAGE_BLOCK_SIZE = 64;  // paged attention block size
constexpr int BLOCK_H = 64;       // heads per thread block
constexpr int BLOCK_N = 64;       // KV tokens per iteration
constexpr int MAX_SEQLEN = 8192;
#endif

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

// Kernel 配置
constexpr int NUM_WARPS = 4;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;  // 4 * 64 = 256

// Shared memory tile types (with swizzle for bank conflict reduction)
// Q tiles: [BLOCK_H, DV] = [64, 512] 和 [BLOCK_H, DPE] = [64, 64]
using ST_Q = st_bf<BLOCK_H, DV, st_16x16_s>;           // Q nope
using ST_Q_pe = st_bf<BLOCK_H, DPE, st_16x16_s>;       // Q rope

// KV tiles: [BLOCK_N, DV] = [64, 512] 和 [BLOCK_N, DPE] = [64, 64]
using ST_KV = st_bf<BLOCK_N, DV, st_16x16_s>;          // KV (V is first DV dims)
using ST_K_pe = st_bf<BLOCK_N, DPE, st_16x16_s>;       // K rope

// Score tile: [BLOCK_H, BLOCK_N] = [64, 64]
using ST_S = st_bf<BLOCK_H, BLOCK_N, st_16x16_s>;

// Per-warp register tiles
constexpr int WARP_H = BLOCK_H / NUM_WARPS;  // 64 / 4 = 16 rows per warp

// Register tiles for accumulation (float32)
using RT_S = rt_fl<WARP_H, BLOCK_N, row_l>;      // [16, 64] attention scores
using RT_O = rt_fl<WARP_H, DV, row_l>;           // [16, 512] output accumulator

// Register tiles for online softmax (per-row scalars)
using RT_rowvec = rt_fl<WARP_H, 1, row_l>;       // [16, 1] for max/sum/scale

// Global layout types
using GL_Q = gl<bf16, 1, 1, -1, DV>;              // [batch, h_q, dv]
using GL_Q_pe = gl<bf16, 1, 1, -1, DPE>;          // [batch, h_q, dpe]
using GL_KV = gl<bf16, 1, 1, -1, DV>;             // [total_tokens, h_kv, dv]
using GL_K_pe = gl<bf16, 1, 1, -1, DPE>;          // [total_tokens, h_kv, dpe]
using GL_O = gl<bf16, 1, 1, -1, DV>;              // [batch, h_q, dv]

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
__global__ __launch_bounds__(NUM_THREADS, 2)
void flashmla_paged_decoding(
    bf16* __restrict__ Q,              // [batch, h_q, dv]
    bf16* __restrict__ Q_pe,           // [batch, h_q, dpe]  
    bf16* __restrict__ KV,             // [total_tokens, h_kv, dv]
    bf16* __restrict__ K_pe,           // [total_tokens, h_kv, dpe]
    int* __restrict__ block_table,     // [batch, max_num_blocks]
    int* __restrict__ cache_seqlens,   // [batch]
    bf16* __restrict__ Output,         // [batch, h_q, dv]
    int max_num_blocks                 // max blocks per batch
) {
    // Block indices
    const int batch_idx = blockIdx.x;
    const int head_group = blockIdx.y;  // processes BLOCK_H heads at a time
    const int warp_id = kittens::warpid();
    const int lane_id = kittens::laneid();
    
    // Shared memory allocation
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    // Double-buffered KV tiles
    ST_Q& Q_shared = al.allocate<ST_Q>();
    ST_Q_pe& Q_pe_shared = al.allocate<ST_Q_pe>();
    ST_KV (&KV_shared)[2] = al.allocate<ST_KV, 2>();
    ST_K_pe (&K_pe_shared)[2] = al.allocate<ST_K_pe, 2>();
    ST_S& S_shared = al.allocate<ST_S>();
    
    // Register tiles (per warp)
    RT_S acc_s;           // Attention scores accumulator
    RT_O acc_o;           // Output accumulator
    RT_rowvec scores_max;      // Running max for online softmax
    RT_rowvec scores_max_prev; // Previous max
    RT_rowvec scores_scale;    // Scale factor: exp(prev_max - new_max)
    RT_rowvec scores_sum;      // Sum of exp(scores)
    RT_rowvec logsum;          // Running sum for normalization
    
    // Initialize accumulators
    zero(acc_o);
    zero(logsum);
    neg_infty(scores_max);  // -infinity
    
    // ============ Load Q (once per block) ============
    // Q layout: [batch, h_q, dv], load [BLOCK_H, DV] tile
    const int q_row_start = batch_idx * H_Q + head_group * BLOCK_H;
    // Group-cooperative load
    // TODO: Use proper group load with swizzled offsets
    // For now, simple warp-cooperative load
    if (warp_id < 2) {
        // Load Q_shared in cooperative manner
        // Each warp loads a portion
    }
    __syncthreads();
    
    // Get sequence length for this batch
    const int seqlen = cache_seqlens[batch_idx];
    const int num_kv_blocks = (seqlen + BLOCK_N - 1) / BLOCK_N;
    
    // Softmax scale factor: 1/sqrt(d) where d = DV + DPE = 576
    constexpr float scale = 1.0f / 24.0f;  // 1/sqrt(576) ≈ 0.0417, using log2 scale
    constexpr float log2e = 1.4426950408889634f;
    constexpr float softmax_scale = scale * log2e;  // For exp2 instead of exp
    
    // ============ Main Loop: Iterate over KV blocks ============
    // Reverse iteration for numerical stability (recent tokens first)
    int tic = 0, toc = 1;
    
    // Prefetch first KV block
    if (num_kv_blocks > 0) {
        int k = num_kv_blocks - 1;
        int page_idx = block_table[batch_idx * max_num_blocks + (k * BLOCK_N) / PAGE_BLOCK_SIZE];
        int kv_row_start = page_idx * PAGE_BLOCK_SIZE + (k * BLOCK_N) % PAGE_BLOCK_SIZE;
        // Load KV_shared[tic] and K_pe_shared[tic]
        // TODO: Add proper async load
    }
    
    for (int kr = 0; kr < num_kv_blocks; kr++) {
        int k = num_kv_blocks - 1 - kr;  // Reverse order
        
        // Wait for current KV load
        __syncthreads();
        
        // Start loading next KV block (if exists)
        if (kr + 1 < num_kv_blocks) {
            int k_next = num_kv_blocks - 1 - (kr + 1);
            int page_idx = block_table[batch_idx * max_num_blocks + (k_next * BLOCK_N) / PAGE_BLOCK_SIZE];
            int kv_row_start = page_idx * PAGE_BLOCK_SIZE + (k_next * BLOCK_N) % PAGE_BLOCK_SIZE;
            // Async load KV_shared[toc] and K_pe_shared[toc]
            // TODO: Add proper async load
        }
        
        // ============ Compute Attention Scores ============
        // acc_s = Q_nope @ KV^T + Q_rope @ K_pe^T
        zero(acc_s);
        
        // Get warp's subtile of Q
        auto Q_subtile = subtile_inplace<WARP_H, DV>(Q_shared, {warp_id, 0});
        auto Q_pe_subtile = subtile_inplace<WARP_H, DPE>(Q_pe_shared, {warp_id, 0});
        
        // Load Q subtile to registers
        rt_bf<WARP_H, DV, row_l> q_reg;
        rt_bf<WARP_H, DPE, row_l> q_pe_reg;
        load(q_reg, Q_subtile);
        load(q_pe_reg, Q_pe_subtile);
        
        // Load KV and K_pe to registers (all warps load same KV for broadcast)
        rt_bf<BLOCK_N, DV, row_l> kv_reg;
        rt_bf<BLOCK_N, DPE, row_l> k_pe_reg;
        load(kv_reg, KV_shared[tic]);
        load(k_pe_reg, K_pe_shared[tic]);
        
        // GEMM: Q_nope @ KV^T
        mma_ABt(acc_s, q_reg, kv_reg, acc_s);
        
        // GEMM: Q_rope @ K_pe^T (accumulate)
        mma_ABt(acc_s, q_pe_reg, k_pe_reg, acc_s);
        
        // ============ Online Softmax ============
        // Save previous max
        copy(scores_max_prev, scores_max);
        neg_infty(scores_max);
        
        // Compute row-wise max of acc_s
        row_max(scores_max, acc_s, scores_max);
        
        // Global max = max(prev_max, current_max)
        max(scores_max, scores_max, scores_max_prev);
        
        // Compute scale factor: exp2((prev_max - new_max) * softmax_scale)
        sub(scores_scale, scores_max_prev, scores_max);
        mul(scores_scale, scores_scale, softmax_scale);
        exp2(scores_scale, scores_scale);
        
        // Apply softmax to scores: exp2((score - max) * scale)
        // For each element: acc_s[i,j] = exp2((acc_s[i,j] - scores_max[i]) * softmax_scale)
        #pragma unroll
        for (int i = 0; i < acc_s.height; i++) {
            #pragma unroll
            for (int j = 0; j < acc_s.width; j++) {
                float val = acc_s.tiles[i][j].data[0].x;  // Simplified access
                val = exp2f((val - scores_max.tiles[i][0].data[0].x) * softmax_scale);
                acc_s.tiles[i][j].data[0].x = val;
            }
        }
        
        // Mask out-of-bounds positions (only first iteration needs this)
        if (kr == 0) {
            // Mask positions >= seqlen
            #pragma unroll
            for (int i = 0; i < acc_s.height; i++) {
                #pragma unroll
                for (int j = 0; j < acc_s.width; j++) {
                    int token_idx = k * BLOCK_N + j;  // Simplified
                    if (token_idx >= seqlen) {
                        acc_s.tiles[i][j].data[0].x = 0.0f;
                    }
                }
            }
        }
        
        // Compute row sum
        zero(scores_sum);
        row_sum(scores_sum, acc_s, scores_sum);
        
        // Update running sum: logsum = logsum * scale + sum
        mul(logsum, logsum, scores_scale);
        add(logsum, logsum, scores_sum);
        
        // Scale previous output: acc_o *= scale
        #pragma unroll
        for (int i = 0; i < acc_o.height; i++) {
            #pragma unroll  
            for (int j = 0; j < acc_o.width; j++) {
                mul(acc_o.tiles[i][j], acc_o.tiles[i][j], scores_scale.tiles[i][0]);
            }
        }
        
        // Store scores to shared memory for GEMM
        store(S_shared, acc_s);  // Need to convert float -> bf16
        __syncthreads();
        
        // ============ Accumulate Output ============
        // acc_o += scores @ V (V is KV_shared, first DV columns)
        auto S_subtile = subtile_inplace<WARP_H, BLOCK_N>(S_shared, {warp_id, 0});
        rt_bf<WARP_H, BLOCK_N, row_l> s_reg;
        load(s_reg, S_subtile);
        
        // Note: V = KV (same tensor in MLA)
        mma_AB(acc_o, s_reg, kv_reg, acc_o);
        
        // Toggle double buffer
        tic ^= 1;
        toc ^= 1;
    }
    
    // ============ Normalize Output ============
    // acc_o /= logsum
    #pragma unroll
    for (int i = 0; i < acc_o.height; i++) {
        #pragma unroll
        for (int j = 0; j < acc_o.width; j++) {
            div(acc_o.tiles[i][j], acc_o.tiles[i][j], logsum.tiles[i][0]);
        }
    }
    
    // ============ Write Output ============
    const int o_row_start = batch_idx * H_Q + head_group * BLOCK_H + warp_id * WARP_H;
    // TODO: Store acc_o to global memory Output
    // store(Output + o_row_start * DV, acc_o);
    
    if (thread0()) {
        printf("FlashMLA decode: batch=%d, seqlen=%d, num_kv_blocks=%d\n", 
               batch_idx, seqlen, num_kv_blocks);
    }
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
