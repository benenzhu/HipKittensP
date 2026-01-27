# Copyright (c) 2024 Doraemonzzz

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    Out_debug, # debug dump
    kv0_debug,
    kv1_debug,
    S,  # log lambda
    b__1: tl.constexpr,
    h__1: tl.constexpr,
    n__64: tl.constexpr,
    d__128: tl.constexpr,
    e__128: tl.constexpr,
    BLOCK__64: tl.constexpr,
    NUM_BLOCK__1: tl.constexpr,
    BLOCK_MODEL__32: tl.constexpr,
):
    ##### get offset
    off_bh__0 = tl.program_id(0)
    off_h__0 = off_bh__0 % h__1
    off_e__0_3 = tl.program_id(1)
    qk_offset__0 = off_bh__0 * n__64 * d__128
    v_offset__0 = off_bh__0 * n__64 * e__128
    o_offset__0 = off_bh__0 * n__64 * e__128
    # channel offset
    e_offset__0_96 = off_e__0_3 * BLOCK_MODEL__32

    ##### get block ptr
    Q_block_ptr = Q + qk_offset__0 + tl.arange(0, d__128)[None, :]
    K_trans_block_ptr = K + qk_offset__0 + tl.arange(0, d__128)[:, None]
    V_block_ptr = V + v_offset__0 + e_offset__0_96 + tl.arange(0, BLOCK_MODEL__32)[None, :]
    O_block_ptr = Out + o_offset__0 + e_offset__0_96 + tl.arange(0, BLOCK_MODEL__32)[None, :]
    O_debug_block_ptr = Out_debug + o_offset__0 + e_offset__0_96 + tl.arange(0, BLOCK_MODEL__32)[None, :] # debug dump
    S_block_ptr = S + off_h__0

    ##### init diag decay(Lambda); q, k decay; kv
    s__1 = tl.load(S_block_ptr)
    # q, k decay
    off_block = tl.arange(
        0, BLOCK__64
    )  # Not bug, this is a bit different from algorithm 1, but is mathematically equivalent
    q_decay = tl.exp(-s__1.to(tl.float32) * off_block[:, None])                    # 64x1
    k_trans_decay = tl.exp(-s__1.to(tl.float32) * (BLOCK__64 - off_block[None, :]))    # 1x64
    block_decay = tl.exp(-s__1.to(tl.float32) * BLOCK__64)
    # diag decay
    index = off_block[:, None] - off_block[None, :]
    s_index = s__1 * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)    # 64x64

    # debug o_inter index
    # if off_e == 0:
    #     print(f"off_e {off_e}")
    #     print(f"o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :] {o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]}")
    #     print(o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :] + off_block[:, None] * e)
    #     import pdb
    #     pdb.set_trace()
    # if off_e == 1:
    #     print(f"off_e {off_e}")
    #     print(f"o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :] {o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]}")
    #     print(o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :] + off_block[:, None] * e)
    #     import pdb
    #     pdb.set_trace()
    # if off_e == 2:
    #     print(f"off_e {off_e}")
    #     print(f"o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :] {o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]}")
    #     print(o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :] + off_block[:, None] * e)
    #     import pdb
    #     pdb.set_trace()
    # if off_e == 3:
    #     print(f"off_e {off_e}")
    #     print(f"o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :] {o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]}")
    #     print(o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :] + off_block[:, None] * e)
    #     import pdb
    #     pdb.set_trace()
    
    # kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32) # 128x32
    kv = tl.full([d__128, BLOCK_MODEL__32], value=1.0, dtype=tl.float32)

    # ---- kv debug ptr ----
    # num_e_blocks = e__128 // BLOCK_MODEL__32
    # kv0_ptr = (
    #     kv0_debug
    #     + off_bh * num_e_blocks * d * BLOCK_MODEL
    #     + off_e * d * BLOCK_MODEL
    #     + tl.arange(0, d)[:, None] * BLOCK_MODEL
    #     + tl.arange(0, BLOCK_MODEL)[None, :]
    # )
    # kv1_ptr = (
    #     kv1_debug
    #     + off_bh * num_e_blocks * d * BLOCK_MODEL
    #     + off_e * d * BLOCK_MODEL
    #     + tl.arange(0, d)[:, None] * BLOCK_MODEL
    #     + tl.arange(0, BLOCK_MODEL)[None, :]
    # )
    # if off_e == 0:
    #     print(f"off_e {off_e}")
    #     print(f"off_bh * num_e_blocks * d * BLOCK_MODEL + off_e * d * BLOCK_MODEL {off_bh * num_e_blocks * d * BLOCK_MODEL + off_e * d * BLOCK_MODEL}")
    #     print(tl.arange(0, d)[:, None] * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :])
    #     import pdb
    #     pdb.set_trace()
    # if off_e == 1:
    #     print(f"off_e {off_e}")
    #     print(f"off_bh * num_e_blocks * d * BLOCK_MODEL + off_e * d * BLOCK_MODEL {off_bh * num_e_blocks * d * BLOCK_MODEL + off_e * d * BLOCK_MODEL}")
    #     print(tl.arange(0, d)[:, None] * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :])
    #     import pdb
    #     pdb.set_trace()

    kv0_ptr = (
        kv0_debug
        + off_bh__0 * d__128 * e__128                          # 跳过前面的 batch*head
        + off_e__0_3 * BLOCK_MODEL__32                     # 在 e 维度跳到第 off_e 个 block
        + tl.arange(0, d__128)[:, None] * e__128           # d 维度的 stride
        + tl.arange(0, BLOCK_MODEL__32)[None, :]     # e 维度内的偏移
    )
    kv1_ptr = (
        kv1_debug
        + off_bh__0 * d__128 * e__128                          # 跳过前面的 batch*head
        + off_e__0_3 * BLOCK_MODEL__32                     # 在 e 维度跳到第 off_e 个 block
        + tl.arange(0, d__128)[:, None] * e__128           # d 维度的 stride
        + tl.arange(0, BLOCK_MODEL__32)[None, :]     # e 维度内的偏移
    )
    # if off_e == 0:
    #     print(f"off_e {off_e}")
    #     print(f"off_bh * d * e + off_e * BLOCK_MODEL {off_bh * d * e + off_e * BLOCK_MODEL}")
    #     print(tl.arange(0, d)[:, None] * e + tl.arange(0, BLOCK_MODEL)[None, :])
    #     import pdb
    #     pdb.set_trace()
    # if off_e == 1:
    #     print(f"off_e {off_e}")
    #     print(f"off_bh * d * e + off_e * BLOCK_MODEL {off_bh * d * e + off_e * BLOCK_MODEL}")
    #     print(tl.arange(0, d)[:, None] * e + tl.arange(0, BLOCK_MODEL)[None, :])
    #     import pdb
    #     pdb.set_trace()

    ##### compute
    for i in range(NUM_BLOCK__1):
        # load
        q = tl.load(
            Q_block_ptr + off_block[:, None] * d__128, mask=off_block[:, None] < n__64, other=0.0
        ).to(tl.float32)
        k_trans = tl.load(
            K_trans_block_ptr + off_block[None, :] * d__128,
            mask=off_block[None, :] < n__64,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr + off_block[:, None] * e__128, mask=off_block[:, None] < n__64, other=0.0
        ).to(tl.float32)

        # compute
        qk = tl.dot(q, k_trans) * diag_decay   # q 64x128, k_trans 128x64 -> qk 64x64
        o_intra = tl.dot(qk, v)                # qk 64x64, v 64x32 -> o_intra 64x32
        o_inter = tl.dot(q, kv) * q_decay      # q 64x128, kv 128x32 -> 64x32, q_decay 64x1
        o_inter_raw = tl.dot(q, kv)
        o = o_intra + o_inter
        # for debug
        # o = o_intra

        # save and update
        tl.store(
            O_block_ptr + off_block[:, None] * e__128,
            o.to(O_block_ptr.dtype.element_ty),
            mask=off_block[:, None] < n__64,
        )
        # debug dump
        tl.store(
            O_debug_block_ptr + off_block[:, None] * e__128,
            o_inter_raw.to(O_debug_block_ptr.dtype.element_ty), # dump o_inter
            mask=off_block[:, None] < n__64,
        )

        kv = block_decay * kv + tl.dot(k_trans * k_trans_decay, v)
        if i == 0:
            tl.store(kv0_ptr, kv)
        elif i == 1:
            tl.store(kv1_ptr, kv)
        # import pdb
        # pdb.set_trace()
        off_block += BLOCK__64

@triton.jit
def _fwd_kernel_naive(
    Q,
    K,
    V,
    Out,
    Out_debug, # debug dump
    S,  # log lambda
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    BLOCK_MODEL: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_e = tl.program_id(1)
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    # channel offset
    e_offset = off_e * BLOCK_MODEL

    ##### get block ptr
    Q_block_ptr = Q + qk_offset + tl.arange(0, d)[None, :]
    K_trans_block_ptr = K + qk_offset + tl.arange(0, d)[:, None]
    V_block_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_block_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_debug_block_ptr = Out_debug + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :] # debug dump
    S_block_ptr = S + off_h

    ##### init diag decay(Lambda); q, k decay; kv
    s = tl.load(S_block_ptr)
    # q, k decay
    off_block = tl.arange(
        0, BLOCK
    )  # Not bug, this is a bit different from algorithm 1, but is mathematically equivalent
    q_decay = tl.exp(-s.to(tl.float32) * off_block[:, None])                    # 64x1
    k_trans_decay = tl.exp(-s.to(tl.float32) * (BLOCK - off_block[None, :]))    # 1x64
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)
    # diag decay
    index = off_block[:, None] - off_block[None, :]
    s_index = s * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)    # 64x64
    
    # kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32) # 128x32
    # kv = tl.ones([d, BLOCK_MODEL], dtype=tl.float32) # 128x32
    kv = tl.full([d, BLOCK_MODEL], value=1.0, dtype=tl.float32)
    # import pdb
    # pdb.set_trace()

    ##### compute
    for i in range(NUM_BLOCK):
        # load
        q = tl.load(
            Q_block_ptr + off_block[:, None] * d, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)
        k_trans = tl.load(
            K_trans_block_ptr + off_block[None, :] * d,
            mask=off_block[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr + off_block[:, None] * e, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)

        # compute
        qk = tl.dot(q, k_trans) * diag_decay   # q 64x128, k_trans 128x64 -> qk 64x64
        o_intra = tl.dot(qk, v)                # qk 64x64, v 64x32 -> o_intra 64x32
        o_inter = tl.dot(q, kv) * q_decay      # q 64x128, kv 128x32 -> 64x32, q_decay 64x1
        o_inter_raw = tl.dot(q, kv)
        o = o_intra + o_inter
        # for debug
        # o = o_intra

        # save and update
        tl.store(
            O_block_ptr + off_block[:, None] * e,
            o.to(O_block_ptr.dtype.element_ty),
            mask=off_block[:, None] < n,
        )
        # debug dump
        tl.store(
            O_debug_block_ptr + off_block[:, None] * e,
            o_inter_raw.to(O_debug_block_ptr.dtype.element_ty), # dump o_inter
            mask=off_block[:, None] < n,
        )

        kv = block_decay * kv + tl.dot(k_trans * k_trans_decay, v)
        # import pdb
        # pdb.set_trace()
        off_block += BLOCK


@triton.jit
def _bwd_intra_kernel(
    Q,
    K,
    V,
    S,
    DO,
    DQ,
    DK,
    DV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_h = off_bh % h
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    block_offset = off_block * BLOCK + tl.arange(0, BLOCK)

    ##### get block ptr
    Q_trans_block_ptr = (
        Q + qk_offset + block_offset[None, :] * d + tl.arange(0, d)[:, None]
    )
    K_block_ptr = K + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[None, :]
    V_trans_block_ptr = (
        V + v_offset + block_offset[None, :] * e + tl.arange(0, e)[:, None]
    )

    DQ_block_ptr = DQ + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[None, :]
    DK_trans_block_ptr = (
        DK + qk_offset + block_offset[None, :] * d + tl.arange(0, d)[:, None]
    )
    DV_block_ptr = DV + v_offset + block_offset[:, None] * e + tl.arange(0, e)[None, :]
    DO_block_ptr = DO + o_offset + block_offset[:, None] * e + tl.arange(0, e)[None, :]

    S_block_ptr = S + off_h

    ##### init diag decay(Lambda)
    s = tl.load(S_block_ptr)
    array = tl.arange(0, BLOCK).to(tl.float32)
    # diag
    index = array[:, None] - array[None, :]
    s_index = s * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)
    diag_decay_trans = tl.trans(diag_decay)

    ##### load block
    k = tl.load(K_block_ptr, mask=block_offset[:, None] < n, other=0.0).to(tl.float32)
    v_trans = tl.load(V_trans_block_ptr, mask=block_offset[None, :] < n, other=0.0).to(
        tl.float32
    )
    do = tl.load(DO_block_ptr, mask=block_offset[:, None] < n, other=0.0).to(tl.float32)
    q_trans = tl.load(Q_trans_block_ptr, mask=block_offset[None, :] < n, other=0.0).to(
        tl.float32
    )

    ##### compute
    dqk = tl.dot(do, v_trans) * diag_decay
    dq_intra = tl.dot(dqk, k)

    dk_intra_trans = tl.dot(q_trans, dqk)

    qk_trans = tl.dot(k, q_trans) * diag_decay_trans
    dv_intra = tl.dot(qk_trans, do)

    dq = dq_intra
    dk_trans = dk_intra_trans
    dv = dv_intra

    # save
    tl.store(
        DQ_block_ptr,
        dq.to(DQ_block_ptr.dtype.element_ty),
        mask=block_offset[:, None] < n,
    )
    tl.store(
        DK_trans_block_ptr,
        dk_trans.to(DK_trans_block_ptr.dtype.element_ty),
        mask=block_offset[None, :] < n,
    )
    tl.store(
        DV_block_ptr,
        dv.to(DV_block_ptr.dtype.element_ty),
        mask=block_offset[:, None] < n,
    )


@triton.jit
def _bwd_inter_kernel(
    Q,
    K,
    V,
    S,
    DO,
    DQ,
    DK,
    DV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    S_block_ptr = S + off_h

    ##### get block ptr
    DQ_block_ptr = (
        DQ + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    K_block_ptr = (
        K + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V + v_offset + tl.arange(0, CBLOCK)[None, :] * e + tl.arange(0, e)[:, None]
    )
    DO_block_ptr = (
        DO + o_offset + tl.arange(0, CBLOCK)[:, None] * e + tl.arange(0, e)[None, :]
    )
    # mask
    off_block1 = tl.arange(0, CBLOCK)
    off_block2 = tl.arange(0, CBLOCK)
    # compute block array
    c_array = tl.arange(0, CBLOCK)

    ##### init lambda; kv
    s = tl.load(S_block_ptr)
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)
    kv_trans = tl.zeros([e, d], dtype=tl.float32)

    ##### compute dq inter
    for i in range(NUM_BLOCK):
        # compute in subblock
        for j in range(NUM_CBLOCK):
            if i > 0:  # if not add this, may have bug
                q_decay = tl.exp(-s.to(tl.float32) * (j * CBLOCK + c_array[:, None]))
                do = tl.load(DO_block_ptr, mask=off_block1[:, None] < n, other=0.0).to(
                    tl.float32
                )
                dq_inter = tl.dot(do, kv_trans) * q_decay
                dq = dq_inter + tl.load(
                    DQ_block_ptr, mask=off_block1[:, None] < n, other=0.0
                )
                tl.store(
                    DQ_block_ptr,
                    dq.to(DQ_block_ptr.dtype.element_ty),
                    mask=off_block1[:, None] < n,
                )

            DQ_block_ptr += CBLOCK * d
            DO_block_ptr += CBLOCK * e
            off_block1 += CBLOCK

        # update kv in subblock
        kv_trans_current = tl.zeros([e, d], dtype=tl.float32)
        for j in range(NUM_CBLOCK):
            v_trans = tl.load(
                V_trans_block_ptr, mask=off_block2[None, :] < n, other=0.0
            ).to(tl.float32)
            k = tl.load(K_block_ptr, mask=off_block2[:, None] < n, other=0.0).to(
                tl.float32
            )
            k_decay = tl.exp(
                -s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[:, None]))
            )
            kv_trans_current += tl.dot(v_trans, k * k_decay)

            K_block_ptr += CBLOCK * d
            V_trans_block_ptr += CBLOCK * e
            off_block2 += CBLOCK

        kv_trans = block_decay * kv_trans + kv_trans_current

    ##### get block ptr
    m = NUM_BLOCK * BLOCK
    off_block1 = m + tl.arange(0, CBLOCK)
    off_block2 = m + tl.arange(0, CBLOCK)

    Q_trans_block_ptr = (
        Q
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    K_block_ptr = (
        K
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + m * e
        + tl.arange(0, CBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )

    DK_trans_block_ptr = (
        DK
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    DV_block_ptr = (
        DV
        + v_offset
        + m * e
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + m * e
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    ##### init dkv
    dkv = tl.zeros([d, e], dtype=tl.float32)

    ##### compute dk, dv inter
    for i in range(NUM_BLOCK - 1, -1, -1):
        # compute in subblock
        for j in range(NUM_CBLOCK - 1, -1, -1):
            K_block_ptr -= CBLOCK * d
            V_trans_block_ptr -= CBLOCK * e
            DK_trans_block_ptr -= CBLOCK * d
            DV_block_ptr -= CBLOCK * e
            off_block1 -= CBLOCK

            if i < NUM_BLOCK - 1:  # if not add this, may have bug
                k = tl.load(K_block_ptr, mask=off_block1[:, None] < n, other=0.0).to(
                    tl.float32
                )
                v_trans = tl.load(
                    V_trans_block_ptr, mask=off_block1[None, :] < n, other=0.0
                ).to(tl.float32)

                k_decay_trans = tl.exp(
                    -s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[None, :]))
                )
                k_decay = tl.exp(
                    -s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[:, None]))
                )
                dk_inter_trans = tl.dot(dkv, v_trans) * k_decay_trans
                dv_inter = tl.dot(k, dkv) * k_decay

                dk_trans = dk_inter_trans + tl.load(
                    DK_trans_block_ptr, mask=off_block1[None, :] < n, other=0.0
                )
                dv = dv_inter + tl.load(
                    DV_block_ptr, mask=off_block1[:, None] < n, other=0.0
                )

                tl.store(
                    DK_trans_block_ptr,
                    dk_trans.to(DK_trans_block_ptr.dtype.element_ty),
                    mask=off_block1[None, :] < n,
                )
                tl.store(
                    DV_block_ptr,
                    dv.to(DV_block_ptr.dtype.element_ty),
                    mask=off_block1[:, None] < n,
                )

        # update dkv in subblock
        dkv_current = tl.zeros([d, e], dtype=tl.float32)
        for j in range(NUM_CBLOCK - 1, -1, -1):
            DO_block_ptr -= CBLOCK * e
            Q_trans_block_ptr -= CBLOCK * d
            off_block2 -= CBLOCK

            do = tl.load(DO_block_ptr, mask=off_block2[:, None] < n, other=0.0).to(
                tl.float32
            )
            q_trans = tl.load(
                Q_trans_block_ptr, mask=off_block2[None, :] < n, other=0.0
            ).to(tl.float32)
            q_decay_trans = tl.exp(-s.to(tl.float32) * (j * CBLOCK + c_array[None, :]))
            dkv_current += tl.dot(q_trans * q_decay_trans, do)

        dkv = block_decay * dkv + dkv_current


class LightningAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()
        # print(s)
        print("k.shape: ", k.shape)
        print("v.shape: ", v.shape)
        print("s.shape: ", s.shape)

        b__1, h__1, n__64, d__128 = q.shape
        e__128 = v.shape[-1]
        o = torch.empty((b__1, h__1, n__64, e__128), dtype=q.dtype, device=q.device)
        print("e: ", e__128)

        # debug dump
        o_debug = torch.empty((b__1, h__1, n__64, e__128), dtype=q.dtype, device=q.device)
        kv0_debug = torch.empty((b__1, h__1, d__128, e__128), dtype=torch.float32, device=q.device)
        kv1_debug = torch.empty((b__1, h__1, d__128, e__128), dtype=torch.float32, device=q.device)

        BLOCK__64 = 64
        NUM_BLOCK__1 = triton.cdiv(n__64, BLOCK__64)
        print("NUM_BLOCK: ", NUM_BLOCK__1)
        # parallel over channel
        BLOCK_MODEL__32 = min(triton.next_power_of_2(e__128), 32)
        grid = (b__1 * h__1, triton.cdiv(e__128, BLOCK_MODEL__32))
        
        _fwd_kernel[grid](
            q,
            k,
            v,
            o,
            o_debug,
            kv0_debug,
            kv1_debug,
            s,
            b__1,
            h__1,
            n__64,
            d__128,
            e__128,
            BLOCK__64=BLOCK__64,
            NUM_BLOCK__1=NUM_BLOCK__1,
            BLOCK_MODEL__32=BLOCK_MODEL__32,
        )

        ctx.save_for_backward(q, k, v, s)

        return o, o_debug, kv0_debug, kv1_debug

    @staticmethod
    def backward(ctx, do):
        q, k, v, s = ctx.saved_tensors

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()
        do = do.contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        b, h, n, d = q.shape
        e = v.shape[-1]

        # block size
        BLOCK = 64
        NUM_BLOCK = triton.cdiv(n, BLOCK)
        # compute block size
        CBLOCK = 32
        NUM_CBLOCK = BLOCK // CBLOCK

        # for intra part, compute in parallel
        grid = (b * h, NUM_BLOCK)
        _bwd_intra_kernel[grid](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        # for inter part, compute in sequencial
        grid = (b * h,)
        _bwd_inter_kernel[grid](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        return dq, dk, dv, None, None


lightning_attn2 = LightningAttention2.apply