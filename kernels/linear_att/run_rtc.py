import os
# os.environ["ROCPROF_COUNTER_COLLECTION"] = "1"
# os.environ["HSA_TOOLS_LIB"]="/opt/rocm/lib/librocm-debug-agent.so.2" 
# os.environ["HSA_ENABLE_DEBUG"]="1o"
os.environ["PYTORCH_NO_HIP_MEMORY_CACHING"]="1"
os.environ["HSA_DISABLE_FRAGMENT_ALLOCATOR"]="1"
os.system("clear")
import torch
from dataclasses import dataclass
import torch
import time
import torch.nn.functional as F
from tqdm import trange
import sys
import numpy as np

import baselines.lightning_attn2_dump
import baselines.lightning_attn2_naive
import importlib
importlib.reload(baselines.lightning_attn2_dump)
importlib.reload(baselines.lightning_attn2_naive)
import os
os.system("clear")
import rtc
importlib.reload(rtc)
from rtc import _compile_kernel, get_triton_gemm_NTN, my_assert_close, log, get_kernel
torch.set_printoptions(threshold=1000, edgeitems=10, sci_mode=False, precision=6, linewidth=300)     

# from baselines.lightning_attn2 import lightning_attn2
from baselines.lightning_attn2_dump import lightning_attn2
from baselines.lightning_attn2_naive import lightning_attn2_naive

D_QK = 128
D_VO = 128
CHUNK_SIZE = 64


def generate_inputs(B, H, N):
    # q = torch.arange(B * H * N * D_QK, dtype=torch.bfloat16, device='cuda').reshape(B, H, N, D_QK) * 0.00005 + 0.5
    # k = torch.arange(B * H * N * D_QK, dtype=torch.bfloat16, device='cuda').reshape(B, H, N, D_QK) * 0.00005
    q = torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda') / (D_QK ** 0.5)
    k = torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda') / (D_QK ** 0.5)
    v = torch.randn((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda')
    # v = torch.arange(B * H * N * D_VO, dtype=torch.bfloat16, device='cuda').reshape(B, H, N, D_VO) * 0.00005

    # q = torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')
    # k = torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')
    # k[:,:,0:N//2,:] = k[:,:,0:N//2,:] / 2
    # k[:,:,N//2:N,:] = k[:,:,N//2:N,:] / 4
    # v = torch.ones((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda')
    # v[:,:,:,0:D_VO//2] = v[:,:,:,0:D_VO//2] / 4
    # v[:,:,:,D_VO//2:D_VO] = v[:,:,:,D_VO//2:D_VO] / 2

    # q = torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')
    # k = torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda') * 2
    # v = torch.ones((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda') * 3
    
    s = torch.rand((H,), dtype=torch.float32, device='cuda')  # s stays float32
    return q, k, v, s

def get_mask(n, slope=1):
    mask = torch.triu(torch.zeros(n, n).float().fill_(float("-inf")), 1)
    for i in range(n):
        x = torch.arange(i + 1)
        y = slope * x
        mask[i, :i + 1] = -torch.flip(y, [0])
    return torch.exp(mask)

def get_full_mask(n, slopes):
    arr = []
    for slope in slopes:
        arr.append(get_mask(n, slope.item()))
    mask = torch.stack(arr, dim=0)
    return mask

def linear_attn(q, k, v, s):
    b, h, n, d = q.shape
    mask = get_full_mask(n, s).to(q.device).to(torch.float32)
    qk = torch.matmul(q, k.transpose(2, 3))
    qk = (qk.to(torch.float32) * mask).to(q.dtype)
    o = torch.matmul(qk, v)
    return o

def linear_attn_naive_qkv(q, k, v):
    qk = torch.matmul(q, k.transpose(2, 3))
    o = torch.matmul(qk, v)
    return o

def linear_attn_naive_qkv_lightning_version(q, k, v):
    b, h, n, d = q.shape
    num_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    o = torch.empty_like(q)

    for i in range(num_chunks):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n)
        q_chunk = q[:, :, start:end, :]
        k_chunk = k[:, :, start:end, :]
        v_chunk = v[:, :, start:end, :]
    
        qk = torch.matmul(q_chunk, k_chunk.transpose(2, 3))
        o_chunk = torch.matmul(qk, v_chunk)
        o[:, :, start:end, :] = o_chunk
    return o

def linear_attn_ref_online(q___1_1_64_128, k__1_1_64_128, v__1_1_64_128, s, block__64=CHUNK_SIZE, block_model__32=None, return_debug=False):
    """
    Torch reference for lightning_attn2 forward.
    Mirrors the Triton _fwd_kernel in baselines/lightning_attn2_dump.py.
    The e dimension is processed in block_model tiles to match kernel grid.
    """
    b__1, h__1, n__64, d__128 = q___1_1_64_128.shape
    print(n__64, d__128)
    e__128 = v__1_1_64_128.shape[-1]
    print(v__1_1_64_128.shape)
    if block_model__32 is None:
        block_model__32 = min(1 << (e__128 - 1).bit_length(), 32)
    num_blocks__1 = (n__64 + block__64 - 1) // block__64
    print(block_model__32)
    print(block__64)

    out = torch.zeros((b__1, h__1, n__64, e__128), dtype=q___1_1_64_128.dtype, device=q___1_1_64_128.device)
    out_debug = torch.zeros_like(out) if return_debug else None
    kv0_debug = torch.zeros((b__1, h__1, d__128, e__128), dtype=torch.float32, device=q___1_1_64_128.device) if return_debug else None
    kv1_debug = torch.zeros((b__1, h__1, d__128, e__128), dtype=torch.float32, device=q___1_1_64_128.device) if return_debug else None

    off = torch.arange(block__64, device=q___1_1_64_128.device, dtype=torch.float32)
    index = off[:, None] - off[None, :]
    print(index)
    # print(v__1_1_64_p128.shape)
    # 1/0

    if s.ndim not in (1, 2):
        raise ValueError(f"Expected s to have shape [H] or [B, H], got {tuple(s.shape)}")

    for bi in range(b__1):
        for hi in range(h__1):
            s_val = s[hi] if s.ndim == 1 else s[bi, hi]
            s_val = s_val.to(torch.float32)
            q_decay_full = torch.exp(-s_val * off)
            k_trans_decay_full = torch.exp(-s_val * (block__64 - off))
            block_decay = torch.exp(-s_val * block__64)
            diag_decay_full = torch.where(
                index >= 0,
                torch.exp(-s_val * index),
                torch.zeros_like(index),
            )
            # print(f"diag_decay_full: {diag_decay_full}")
            # 1/0
            

            # for e_start in range(0, e__128, block_model__32): ## 4次
            for e_start in range(0, 1): ## 4次
            
                e_end = min(e_start + block_model__32, e__128)
                # print(f"e_start: {e_start}, e_end: {e_end}")
                kv__128_32 = torch.full((d__128, e_end - e_start), 1.0, dtype=torch.float32, device=q___1_1_64_128.device)

                for blk in range(num_blocks__1):
                    start__0 = blk * block__64
                    end__64 = min(start__0 + block__64, n__64)
                    if start__0 >= end__64:
                        break
                    length = end__64 - start__0

                    q_reg = q___1_1_64_128[bi, hi, start__0:end__64, :].to(torch.float32)
                    k_reg = k__1_1_64_128[bi, hi, start__0:end__64, :].to(torch.float32)
                    v_blk__64_32 = v__1_1_64_128[bi, hi, start__0:end__64, e_start:e_end].to(torch.float32)

                    diag_decay = diag_decay_full[:length, :length]
                    print("diag_decay\n", diag_decay)
                    q_decay__64_1 = q_decay_full[:length].unsqueeze(-1)
                    k_trans_decay = k_trans_decay_full[:length]



                    #   O = ((Q @ KT) * diag) @ v + (Q * q_decay) @ kvstate

                    print("att_block_origin.T\n", torch.matmul(q_reg, k_reg.transpose(0, 1)).T)
                    # TODOXUN
                    # qk__64_64 = torch.matmul(q_reg, k_reg.transpose(0, 1)) * diag_decay
                    qk__64_64 = torch.matmul(q_reg, k_reg.transpose(0, 1))
                    print("att_block.T\n", qk__64_64.T)
                    #  3. s @ v
                    o_intra__64_32 = torch.matmul(qk__64_64, v_blk__64_32)
                    print("o_intra_64_32.T\n", o_intra__64_32.T)
                    print("o_intra_64_32.T\n", (o_intra__64_32.T[:,32:]))
                    o_inter_raw__64_32 = torch.matmul(q_reg, kv__128_32)
                    
                    print("o_inter.T\n", o_inter_raw__64_32.T)
                    print("o_inter[32:].T\n", (o_inter_raw__64_32.T[:,32:]))
                    o_inter = o_inter_raw__64_32 # * q_decay__64_1
                    o_blk__64_32 = o_intra__64_32 + o_inter
                    # print(f"o_inter_raw.shape: {o_inter_rpaw__64_32.shape}", kv__128_32.shape)
                    # print(f"q_decay.shape: {q_decay__64_1.shape}")
                    # print(f"o_intra.shape: {o_intra__64_32.shape}")


                    out[bi, hi, start__0:end__64, e_start:e_end] = o_blk__64_32.to(out.dtype)
                    if return_debug:
                        out_debug[bi, hi, start__0:end__64, e_start:e_end] = o_inter_raw__64_32.to(out.dtype)

                    kv__128_32 = block_decay * kv__128_32 + torch.matmul(
                        k_reg.transpose(0, 1) * k_trans_decay, v_blk__64_32
                    )
                    if return_debug:
                        if blk == 0:
                            kv0_debug[bi, hi, :, e_start:e_end] = kv__128_32
                        elif blk == 1:
                            kv1_debug[bi, hi, :, e_start:e_end] = kv__128_32

    if return_debug:
        return out, out_debug, kv0_debug, kv1_debug
    return out

def save_test_case(q, k, v, s, o, n, o_debug=None):
    filename = f'naive_qkv_randn_{n}.txt'
    print(f"slopes: {s}")
    import pdb
    # pdb.set_trace()
    with open(filename, 'w') as f:    
        sf = s.to(torch.float32).flatten().cpu().numpy().tolist()
        qf = q.to(torch.float32).flatten().cpu().numpy().tolist()
        kf = k.to(torch.float32).flatten().cpu().numpy().tolist()
        vf = v.to(torch.float32).flatten().cpu().numpy().tolist()
        of = o.to(torch.float32).flatten().cpu().numpy().tolist()

        for i in trange(len(sf)):
            f.write(repr(sf[i]))
            f.write(' ')

        for i in trange(len(qf)):
            f.write(repr(qf[i]))
            f.write(' ')
            
        for i in trange(len(kf)):
            f.write(repr(kf[i]))
            f.write(' ')

        for i in trange(len(vf)):
            f.write(repr(vf[i]))
            f.write(' ')

        for i in trange(len(of)):
            f.write(repr(of[i]))
            f.write(' ')
        
        if o_debug is not None:
            o_debug_f = o_debug.to(torch.float32).flatten().cpu().numpy().tolist()
            for i in trange(len(o_debug_f)):
                f.write(repr(o_debug_f[i]))
                f.write(' ')


torch.manual_seed(0)
# torch.manual_seed(42)

# B, H = 16, 8
# sequence_lengths = [1024]

B, H = 1, 1
sequence_lengths = [64]
# sequence_lengths = [128]
# sequence_lengths = [1024]


def run_kittens_mla(q, k, v, s, out):
    mla_kittens = get_kernel("lightning_attn2_kernel", "linear.cpp")  
    grid = (1, 1, 1)
    block = (512, 1, 1)
    out = torch.zeros_like(out)
    
    print("q.shape", q.shape, "k.shape", k.shape, "v.shape", v.shape, "s.shape", s.shape, "out.shape", out.shape)
    print("q.ptr", hex(q.data_ptr()), "k.ptr", hex(k.data_ptr()), "v.ptr", hex(v.data_ptr()), "out.ptr", hex(out.data_ptr()))
    mla_kittens(grid, block, (q, k, v, s, out), shared_mem=160000)

    return out
for N in sequence_lengths:
    print(f"\nGenerating test case for sequence length {N}")
    q, k, v, s = generate_inputs(B, H, N)

    # pytorch_out = linear_attn(q, k, v, s)
    # pytorch_out = linear_attn_naive_qkv(q, k, v)
    # pytorch_out = linear_attn_naive_qkv_lightning_version(q, k, v)
    pytorch_out, pytorch_debug_out, kv0_ref, kv1_ref = linear_attn_ref_online(
        q, k, v, s, return_debug=True
    )
    import pdb
    # pdb.set_trace()
    triton_out, triton_debug_out, kv0_debug, kv1_debug = lightning_attn2(q, k, v, s)
    triton_out_naive, triton_debug_out_naive, kv0_naive, kv1_naive = lightning_attn2_naive(q, k, v, s)
    # print(f"kv0_naive: {kv0_naive}")
    # print(f"kv1_naive: {kv1_naive}")
    # print(f"kv0_debug: {kv0_debug}")
    # print(f"kv1_debug: {kv1_debug}")
    # np.save("kv0_naive.npy", kv0_naive.cpu().numpy())
    # np.save("kv1_naive.npy", kv1_naive.cpu().numpy())
    # np.save("triton_out_naive_block32.npy", triton_out_naive.to(torch.float32).cpu().numpy())
    # pdb.set_trace()
    assert torch.allclose(triton_out, triton_out_naive, atol=1e-3, rtol=1e-3)
    assert torch.allclose(triton_debug_out, triton_debug_out_naive, atol=1e-3, rtol=1e-3)
    assert torch.allclose(kv0_debug, kv0_naive, atol=1e-5, rtol=1e-5)
    # """ assert torch.allclose """(pytorch_out, triton_out, atol=1e-5, rtol=1e-5)
    # assert torch.allclose(kv1_debug, kv1_naive, atol=1e-5, rtol=1e-5)
    
    avg_mag_pytorch = torch.mean(torch.abs(pytorch_out)).item()
    # avg_mag_triton = torch.mean(torch.abs(triton_out)).item()
    # max_diff = torch.max(torch.abs(pytorch_out - triton_out)).item()
    # avg_diff = torch.mean(torch.abs(pytorch_out - triton_out)).item()
    # assert torch.allclose(pytorch_out, triton_out, atol=1e-3, rtol=1e-3)
    
    print(f"PyTorch output magnitude: {avg_mag_pytorch}")
    # print(f"Triton  output magnitude: {avg_mag_triton}")
    # print(f"Max     difference between PyTorch and Triton: {max_diff}")
    # print(f"Average difference between PyTorch and Triton: {avg_diff}")
    
    # save_test_case(q, k, v, s, triton_out, N)
    # save_test_case(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), s, pytorch_out.transpose(1, 2), N) # for amd layout
    # save_test_case(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), s, triton_out.transpose(1, 2), N) # for amd layout
    # debug dump
    # save_test_case(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), s, triton_out.transpose(1, 2), N, triton_debug_out.transpose(1, 2)) # for amd layout
    print(f"Generated random test case for N={N}")
    print(q.shape, k.shape, v.shape, s.shape, triton_out.shape)
    kittens_out = run_kittens_mla(q, k, v, s, triton_out)
    # print(f"{kittens_out=}")
    # print(f"{pytorch_out=}")
    abs_diff = torch.abs(kittens_out - pytorch_out)
    rel_diff = abs_diff / (torch.abs(pytorch_out) + 1e-12)
    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    print(f"max_abs_diff: {max_abs_diff:.6g}, max_rel_diff: {max_rel_diff:.6g}")
    assert torch.allclose(kittens_out, pytorch_out, atol=1e-3, rtol=1e-3)



