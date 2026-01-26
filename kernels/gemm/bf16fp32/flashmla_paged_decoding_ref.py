import math
from typing import Dict, List, Optional, Tuple

import torch


def _exp(x: torch.Tensor, use_exp2: bool) -> torch.Tensor:
    if use_exp2:
        if hasattr(torch, "exp2"):
            return torch.exp2(x)
        return torch.pow(2.0, x)
    return torch.exp(x)


def _validate_shapes(
    q: torch.Tensor,
    kv: torch.Tensor,
    qpe: Optional[torch.Tensor],
    kvpe: Optional[torch.Tensor],
    include_pe: bool,
) -> None:
    if q.dim() != 3:
        raise ValueError(f"q must be [B, H_Q, D], got {q.shape}")
    if kv.dim() != 3:
        raise ValueError(f"kv must be [B, S, D], got {kv.shape}")
    if q.shape[0] != kv.shape[0]:
        raise ValueError("q and kv must have the same batch size")
    if include_pe:
        if (qpe is None) != (kvpe is None):
            raise ValueError("qpe and kvpe must be both set when include_pe is True")
        if qpe is not None:
            if qpe.dim() != 3 or qpe.shape[:2] != q.shape[:2]:
                raise ValueError(f"qpe must be [B, H_Q, DPE], got {qpe.shape}")
        if kvpe is not None:
            if kvpe.dim() != 3 or kvpe.shape[:2] != kv.shape[:2]:
                raise ValueError(f"kvpe must be [B, S, DPE], got {kvpe.shape}")


def _slice_q_heads(
    q: torch.Tensor,
    block_h: int,
    head_group: Optional[int],
) -> torch.Tensor:
    if head_group is None:
        return q
    head_start = head_group * block_h
    head_end = head_start + block_h
    if head_end > q.shape[1]:
        raise ValueError("head_group * block_h exceeds H_Q")
    return q[:, head_start:head_end]


def flashmla_ref_full(
    q: torch.Tensor,
    kv: torch.Tensor,
    qpe: Optional[torch.Tensor] = None,
    kvpe: Optional[torch.Tensor] = None,
    block_h: int = 64,
    head_group: Optional[int] = None,
    scale: Optional[float] = None,
    use_exp2: bool = True,
    include_pe: bool = True,
) -> torch.Tensor:
    """
    Full attention reference for kernel shapes:
    q  : [B, H_Q, DV]
    kv : [B, S, DV]
    qpe: [B, H_Q, DPE] (optional)
    kvpe: [B, S, DPE]  (optional)
    """
    _validate_shapes(q, kv, qpe, kvpe, include_pe)
    q_sel = _slice_q_heads(q, block_h, head_group)
    print(f"{q_sel.shape=}")
    qf = q_sel
    kvf = kv

    scores = torch.matmul(qf, kvf.transpose(-1, -2)).float()
    if include_pe:
        qpe_sel = _slice_q_heads(qpe, block_h, head_group)
        scores = scores + torch.matmul(qpe_sel.float(), kvpe.float().transpose(-1, -2))
    if scale is not None:
        scores = scores * scale

    maxv = scores.max(dim=-1, keepdim=True).values
    probs = _exp(scores - maxv, use_exp2)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    out = torch.matmul(probs.bfloat16(), kvf).float()
    return out


def flashmla_ref_online(
    q: torch.Tensor,
    kv: torch.Tensor,
    qpe: Optional[torch.Tensor] = None,
    kvpe: Optional[torch.Tensor] = None,
    block_n: int = 64,
    block_h: int = 64,
    head_group: Optional[int] = None,
    scale: Optional[float] = None,
    use_exp2: bool = True,
    return_debug: bool = False,
    include_pe: bool = True,
) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
    """
    Blocked online softmax reference. Mirrors the kernel's structure:
    - maintains max (m) and sum (l) per row
    - accumulates unnormalized output and divides by l at the end
    """
    _validate_shapes(q, kv, qpe, kvpe, include_pe)
    q_sel = _slice_q_heads(q, block_h, head_group)
    shared_Q = q_sel
    kvf = kv

    if include_pe and qpe is not None and kvpe is not None:
        qpe_sel = _slice_q_heads(qpe, block_h, head_group)
        qpe_f = qpe_sel
        kvpe_f = kvpe
    else:
        qpe_f = None
        kvpe_f = None

    bsz, h_q, _ = shared_Q.shape
    seq_len = kvf.shape[1]
    max_vec = torch.full((bsz, h_q), -float("inf"), device=shared_Q.device, dtype=torch.float32)
    l = torch.zeros((bsz, h_q), device=shared_Q.device, dtype=torch.float32)
    acc_o = torch.zeros((bsz, h_q, kvf.shape[2]), device=shared_Q.device, dtype=torch.float32)

    debug: Optional[List[Dict[str, torch.Tensor]]] = [] if return_debug else None

    for start in range(0, seq_len, block_n):
        end = min(start + block_n, seq_len)
        shared_KV = kvf[:, start:end]
        acc_s = torch.matmul(shared_Q, shared_KV.transpose(-1, -2)).float()
        if qpe_f is not None and kvpe_f is not None:
            acc_s = acc_s + torch.matmul(qpe_f, kvpe_f[:, start:end].transpose(-1, -2))
        if scale is not None:
            acc_s = acc_s * scale

        block_max = acc_s.max(dim=-1).values
        max_vec_new = torch.maximum(max_vec, block_max)

        scale_prev = _exp(max_vec - max_vec_new, use_exp2)

        acc_s_trans = acc_s - max_vec_new.unsqueeze(-1)
        acc_s_trans = _exp(acc_s_trans, use_exp2)


        l = l * scale_prev + acc_s_trans.sum(dim=-1)
        acc_o = acc_o * scale_prev.unsqueeze(-1) + torch.matmul(p.bfloat16(), shared_KV)

        if return_debug and debug is not None:
            debug.append(
                {
                    "start": torch.tensor(start, device=shared_Q.device),
                    "block_max": block_max,
                    "m_new": max_vec_new,
                    "scale_prev": scale_prev,
                    "block_sum": p.sum(dim=-1),
                }
            )

        max_vec = max_vec_new

    out = acc_o / l.clamp_min(1e-20).unsqueeze(-1)
    return out, debug


def compare_refs(
    q: torch.Tensor,
    kv: torch.Tensor,
    qpe: Optional[torch.Tensor] = None,
    kvpe: Optional[torch.Tensor] = None,
    block_n: int = 64,
    block_h: int = 64,
    head_group: Optional[int] = None,
    scale: Optional[float] = None,
    use_exp2: bool = True,
    include_pe: bool = True,
) -> Dict[str, float]:
    full = flashmla_ref_full(
        q,
        kv,
        qpe=qpe,
        kvpe=kvpe,
        block_h=block_h,
        head_group=head_group,
        scale=scale,
        use_exp2=use_exp2,
        include_pe=include_pe,
    )
    online, _ = flashmla_ref_online(
        q,
        kv,
        qpe=qpe,
        kvpe=kvpe,
        block_n=block_n,
        block_h=block_h,
        head_group=head_group,
        scale=scale,
        use_exp2=use_exp2,
        return_debug=False,
        include_pe=include_pe,
    )
    diff = (full - online).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "full": full
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bsz, h_q, seq_len, dv = 1, 64, 256, 512
    q = torch.randn(bsz, h_q, dv, device=device, dtype=torch.bfloat16)
    kv = torch.randn(bsz, seq_len, dv, device=device, dtype=torch.bfloat16)
    metrics = compare_refs(
        q,
        kv,
        block_n=64,
        block_h=64,
        head_group=0,
        use_exp2=True,
        include_pe=False,
    )
    print(metrics)
