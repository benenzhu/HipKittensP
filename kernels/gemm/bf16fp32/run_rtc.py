import os
# os.environ["ROCPROF_COUNTER_COLLECTION"] = "1"
# os.environ["HSA_TOOLS_LIB"]="/opt/rocm/lib/librocm-debug-agent.so.2" 
# os.environ["HSA_ENABLE_DEBUG"]="1o"
os.environ["PYTORCH_NO_HIP_MEMORY_CACHING"]="1"
os.environ["HSA_DISABLE_FRAGMENT_ALLOCATOR"]="1"
import torch
from dataclasses import dataclass
import torch
import time
import importlib
import rtc
import os
import math
import flashmla_paged_decoding_ref

importlib.reload(rtc)
importlib.reload(flashmla_paged_decoding_ref)
# import tritonblas
# importlib.reload(tritonblas)
# from tritonblas.matmul import persistent_matmul_lt
# importlib.reload(tritonblas.matmul)
from rtc import _compile_kernel, get_triton_gemm_NTN, my_assert_close, log, get_kernel
from flashmla_paged_decoding_ref import flashmla_ref_full, flashmla_ref_online
torch.set_printoptions(threshold=1000, edgeitems=3, sci_mode=False)     



# Configure shared memory - kernel needs 160KB for double-buffered tiles
# kittens_kernel.set_shared_memory_config(SHARED_MEM_SIZE)

def bench_kernel(fn, M, N, K, warmup=100, rep=500, use_cuda_graph=False):
    """
    Benchmark a kernel function and return TFLOPS.
    
    Args:
        fn: callable that runs the kernel
        M, N, K: matrix dimensions for FLOP calculation
        warmup: number of warmup iterations
        rep: number of benchmark iterations
        use_cuda_graph: whether to capture into CUDA graph to reduce launch overhead
    
    Returns:
        tuple: (latency_ms, tflops)
    """
    from triton.testing import do_bench
    
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    if use_cuda_graph:
        # Capture into CUDA graph
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # Warmup in capture stream
            fn()
        stream.synchronize()
        
        # Capture the graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fn()
        
        # Benchmark the graph replay
        def graph_fn():
            graph.replay()
        
        latency_ms = do_bench(graph_fn, warmup=warmup, rep=rep, return_mode="median")
    else:
        latency_ms = do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    
    # GEMM has 2*M*N*K FLOPs
    tflops = 2 * M * N * K / (latency_ms * 1e-3) * 1e-12
    return latency_ms, tflops


def test_kittens_gemm_kernel(): 
    M, N, K = 2048, 4096, 8192
    A = torch.randn(M, K).cuda().bfloat16().contiguous() * 0.1
    B = torch.randn(N, K).cuda().bfloat16().contiguous() * 0.1
    C = torch.zeros(M, N).cuda().bfloat16().contiguous()
    SHARED_MEM_SIZE = 160000
    kittens_kernel = get_kernel("micro_tk", "256_256_64_32_with16x32_rtc.cpp")
    
    grid = (M // 256 * (N // 256), 1, 1)
    grid = (1,1,1)
    block = (64, 1, 1)
    
    # Define kernel launch function
    def kittens_fn():
        kittens_kernel(grid, block, (A, B, C), shared_mem=SHARED_MEM_SIZE)
    
    # Run once for correctness check
    kittens_fn()
    torch.cuda.synchronize()
    return
    
    # Check correctness
    ref = A @ B.T
    print("C", C)
    print("ref", ref)
    print("mean abs error:", (C - ref).abs().mean().item())
    torch.testing.assert_close(C, ref, atol=1, rtol=1)
    log("Correctness check passed!")
    
    
    C.zero_()  # Reset output
    latency_ms, tflops = bench_kernel(kittens_fn, M, N, K, use_cuda_graph=True)
    log(f"kittens (graph): {tflops:.2f} TFLOPS, {latency_ms:.4f} ms")
    
    latency_ms, tflops = bench_kernel(kittens_fn, M, N, K, use_cuda_graph=False)
    log(f"kittens (no graph): {tflops:.2f} TFLOPS, {latency_ms:.4f} ms")
    
    # Benchmark triton reference
    C_triton = torch.zeros_like(C)
    def triton_fn():
        get_triton_gemm_NTN(A, B, C_triton, M, N, K)
    
    latency_ms, tflops = bench_kernel(triton_fn, M, N, K, use_cuda_graph=False)
    log(f"triton: {tflops:.2f} TFLOPS, {latency_ms:.4f} ms")
    
    return C




def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    """
    标准的 Scaled Dot-Product Attention 实现
    
    Args:
        query: [h_q, s_q, d]
        key: [h_kv, s_kv, d]
        value: [h_kv, s_kv, dv]
        h_q: query head 数量
        h_kv: kv head 数量
        is_causal: 是否使用 causal mask
    
    Returns:
        output: [h_q, s_q, dv]
        lse: [h_q, s_q] log-sum-exp for numerical stability
    """
    query = query.float()
    key = key.float()
    value = value.float()
    # GQA: 复制 KV heads 以匹配 Q heads
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    # QK^T / sqrt(d)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    # Causal mask
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype, device=query.device)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool, device=query.device).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_weight += attn_bias
    # LSE for numerical stability (used in split-kv reduction)
    lse = attn_weight.logsumexp(dim=-1)
    # Softmax
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    # Output
    return attn_weight @ value, lse


@torch.inference_mode()
def ref_mla_decode(q, blocked_kv, block_table, cache_seqlens, h_q, h_kv, d, dv, causal=True):
    """
    MLA Decode 的 PyTorch 参考实现 
    Args:
        q: [b, s_q, h_q, d]  query, d = dv + dpe (576 = 512 + 64)
        blocked_kv: [num_blocks, block_size, h_kv, d]  paged KV cache
        block_table: [b, max_num_blocks]  每个 batch 的 block 索引
        cache_seqlens: [b]  每个 batch 的实际 KV 长度
        h_q: query head 数量 (128)
        h_kv: kv head 数量 (1 for MLA)
        d: query/key head dim (576 = 512 + 64)
        dv: value head dim (512)
        causal: 是否使用 causal mask 
    Returns:
        output: [b, s_q, h_q, dv]
    """
    b, s_q = q.shape[0], q.shape[1]
    block_size = blocked_kv.shape[1]
    max_seqlen_pad = block_table.shape[1] * block_size
    
    # V 只取前 dv 维 (MLA 的核心: K 用全部 d 维, V 只用 dv 维)
    blocked_v = blocked_kv[..., :dv]
    
    out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32, device=q.device)
    
    for i in range(b):
        # 根据 block_table 获取当前 batch 的 KV
        begin = i * max_seqlen_pad
        end = begin + cache_seqlens[i].item()
        
        # Flatten blocked KV to contiguous KV
        k_seq = blocked_kv.view(-1, h_kv, d)[begin:end]  # [seqlen, h_kv, d]
        v_seq = blocked_v.view(-1, h_kv, dv)[begin:end]  # [seqlen, h_kv, dv]
        
        # Attention
        O, _ = scaled_dot_product_attention(
            q[i].transpose(0, 1),      # [h_q, s_q, d]
            k_seq.transpose(0, 1),     # [h_kv, seqlen, d]
            v_seq.transpose(0, 1),     # [h_kv, seqlen, dv]
            h_q,
            h_kv,
            is_causal=causal,
        )
        out[i] = O.transpose(0, 1)  # [s_q, h_q, dv]
    
    return out

if True:
    # DeepSeek-V2/V3 MLA 参数
    b = 1           # batch size
    s_q__1 = 1         # decode 阶段 query 长度为 1
    h_q__128 = 128       # query heads
    h_kv = 1        # kv heads (MLA absorb 后)
    d__576 = 576         # query/key dim = dv + dpe
    dv__512 = 512        # value dim (kv_lora_rank)
    dpe__64 = 64        # rope dim
    
    block_size = 64
    cache_seqlen = 8192
    randomize_blocks = True  # 设为 True 模拟真实内存碎片
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # 构造输入
    cache_seqlens = torch.tensor([cache_seqlen + i * 100 for i in range(b)], dtype=torch.int32, device=device)
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = math.ceil(max_seqlen / block_size) * block_size
    num_blocks_per_batch = max_seqlen_pad // block_size
    
    q = torch.randn(b, s_q__1, h_q__128, d__576, dtype=dtype, device=device)
    block_table = torch.randperm(b * num_blocks_per_batch, dtype=torch.int32, device=device).view(b, num_blocks_per_batch)
    blocked_kv = torch.randn(block_table.numel(), block_size, h_kv, d__576, dtype=dtype, device=device)
    
    # 运行参考实现
    out = ref_mla_decode(q, blocked_kv, block_table, cache_seqlens, h_q__128, h_kv, d__576, dv__512, causal=True)
    
    print(f"Input Q shape: {q.shape}")
    print(f"Input blocked_kv shape: {blocked_kv.shape}")
    print(f"Block table shape: {block_table.shape}")
    print(f"Block table randomized: {randomize_blocks}")
    if randomize_blocks:
        print(f"Block table sample (batch 0): {block_table[0, :5].tolist()}...")
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    print("Reference MLA decode completed successfully!")


def run_kittens_mla(): 
    mla_kittens = get_kernel("flashmla_paged_decoding", "flashmla_paged_decoding.cpp")  
    grid = (1, 1, 1)
    block = (512, 1, 1)
    SEQ_LEN = 4096
    q = torch.randn(b, s_q__1, h_q__128, dv__512).cuda().bfloat16().contiguous() * 0.0 + 1
    qpe = torch.randn(b, s_q__1, h_q__128, dpe__64).cuda().bfloat16().contiguous()
    kv = torch.randn(1, b, SEQ_LEN, dv__512).cuda().bfloat16().contiguous() * 0.0 + 1
    kvpe = torch.randn(1, b, SEQ_LEN, dpe__64).cuda().bfloat16().contiguous()
    out_kernel = torch.zeros((b, s_q__1, h_q__128, dv__512), device="cuda", dtype=torch.bfloat16)
    mla_kittens(grid, block, (q, qpe, kv, kvpe, out_kernel), shared_mem=160000)
    torch.cuda.synchronize()
    print("out") 
    print("out", out_kernel)

    # Reference comparison for debugging
    DEBUG_REF = True
    INCLUDE_PE_REF = False  # kernel currently does not use qpe/kvpe in scores
    USE_ONLINE_REF = False  # set True to use online reference
    HEAD_GROUP = 0
    BLOCK_H = 64
    BLOCK_N = 64

    if DEBUG_REF:
        q_ref = q[:, 0] if q.dim() == 4 else q
        kv_ref = kv[0] if kv.dim() == 4 else kv
        qpe_ref = qpe[:, 0] if INCLUDE_PE_REF and qpe is not None and qpe.dim() == 4 else None
        kvpe_ref = kvpe[0] if INCLUDE_PE_REF and kvpe is not None and kvpe.dim() == 4 else None

        if USE_ONLINE_REF:
            ref_out, _ = flashmla_ref_online(
                q_ref,
                kv_ref,
                qpe=qpe_ref,
                kvpe=kvpe_ref,
                block_n=BLOCK_N,
                block_h=BLOCK_H,
                head_group=HEAD_GROUP,
                include_pe=INCLUDE_PE_REF,
                use_exp2=True,
                return_debug=False,
            )
        out_view = out_kernel[:, 0] if out_kernel.dim() == 4 else out_kernel
        out_slice = out_view[:, HEAD_GROUP * BLOCK_H:(HEAD_GROUP + 1) * BLOCK_H, :].float()
        if False:
            diff = (out_slice - ref_out).abs()
            print(f"[ref] out_slice shape: {tuple(out_slice.shape)}")
            print(f"[ref] max abs diff: {diff.max().item():.6f}")
            print(f"[ref] mean abs diff: {diff.mean().item():.6f}")
    return out_kernel
    
choose = 1
if choose == 0:
    ret = test_kittens_gemm_kernel()
elif choose == 1:
    out = run_kittens_mla()