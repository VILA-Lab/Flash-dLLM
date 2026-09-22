

import math

import torch
from torch import einsum
import triton
import triton.language as tl
import torch.nn.functional as F
from triton.runtime import driver
import torch.nn as nn
# DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _flash_qkv_proj_cache_fwd(
    Xn, Q, # (acc_q_len, d_model)
    K, V, # (acc_k_len, d_model)
    Pos, # (acc_q_len, d_model)
    block_table,
    Wq, Wk, Wv, # (d_model, d_model)
    RotarySin, RotaryCos, # (acc_k_len, head_dim)
    D_MODEL: tl.constexpr,
    HALF: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    num_stages: tl.constexpr,
):
    # step_m = tl.num_programs(0)
    pid_m = tl.program_id(0)  # block over query rows
    pid_h = tl.program_id(1)  # head

    # for m_idx in tl.range(pid_m, NUM_BLOCKS, step_m, num_stages=num_stages):
    bt_row = block_table + pid_m * 4
    start_m = tl.load(bt_row + 2)
    end_m = tl.load(bt_row + 3)

    range_half = tl.arange(0, HALF)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_h = pid_h * HEAD_DIM + range_half
    offs_d = tl.arange(0, BLOCK_D)

    q_mask = offs_m < end_m
    pos = tl.load(Pos + offs_m, mask=q_mask, other=0.0)
    
    offs_q = offs_m[:, None] * D_MODEL + offs_h[None, :]
    offs_k = pos[:, None] * D_MODEL + offs_h[None, :]
    offs_x = offs_m[:, None] * D_MODEL + offs_d[None, :]

    offs_w1 = offs_h[None, :] * D_MODEL + offs_d[:, None]
    offs_w2 = (offs_h + HALF)[None, :] * D_MODEL + offs_d[:, None]

    # QKV projection for a single head
    acc_q1 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    acc_k1 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    acc_v1 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)

    acc_q2 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    acc_k2 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    acc_v2 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)

    for d in range(0, D_MODEL, BLOCK_D):
        x = tl.load(Xn + offs_x + d, mask=q_mask[:, None], other=0.0)

        wq1 = tl.load(Wq + offs_w1 + d)
        wk1 = tl.load(Wk + offs_w1 + d)
        wv1 = tl.load(Wv + offs_w1 + d)

        wq2 = tl.load(Wq + offs_w2 + d)
        wk2 = tl.load(Wk + offs_w2 + d)
        wv2 = tl.load(Wv + offs_w2 + d)

        acc_q1 += tl.dot(x, wq1)
        acc_k1 += tl.dot(x, wk1)
        acc_v1 += tl.dot(x, wv1)

        acc_q2 += tl.dot(x, wq2)
        acc_k2 += tl.dot(x, wk2)
        acc_v2 += tl.dot(x, wv2)

    # Apply rotary for a single head.
    offs_rot = pos[:, None] * HEAD_DIM + range_half[None, :]
    sin1 = tl.load(RotarySin + offs_rot, mask=q_mask[:, None], other=0.0)
    sin2 = tl.load(RotarySin + offs_rot + HALF, mask=q_mask[:, None], other=0.0)
    cos1 = tl.load(RotaryCos + offs_rot, mask=q_mask[:, None], other=0.0)
    cos2 = tl.load(RotaryCos + offs_rot + HALF, mask=q_mask[:, None], other=0.0)

    rq1 = acc_q1 * cos1 - acc_q2 * sin1
    rq2 = acc_q2 * cos2 + acc_q1 * sin2
    rk1 = acc_k1 * cos1 - acc_k2 * sin1
    rk2 = acc_k2 * cos2 + acc_k1 * sin2

    tl.store(Q + offs_q, rq1, mask=q_mask[:, None])
    tl.store(Q + offs_q + HALF, rq2, mask=q_mask[:, None])
    tl.store(K + offs_k, rk1, mask=q_mask[:, None])
    tl.store(K + offs_k + HALF, rk2, mask=q_mask[:, None])
    tl.store(V + offs_k, acc_v1, mask=q_mask[:, None])
    tl.store(V + offs_k + HALF, acc_v2, mask=q_mask[:, None])


@triton.jit
def _flash_masked_attention_fwd(
    Q, O, # (acc_q_len, d_model)
    K, V, # (acc_k_len, d_model)
    S, # (batch, heads, block_m, max_k_length)
    stride_sm, stride_sh, stride_sb,
    block_table,
    softmax_scale,
    SEQLEN_MAX: tl.constexpr,
    D_MODEL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # block over query rows
    pid_h = tl.program_id(1)  # head

    bt_row = block_table + pid_m * 4
    batch_id = tl.load(bt_row + 0)
    start_n = batch_id * SEQLEN_MAX
    end_n = tl.load(bt_row + 1)
    start_m = tl.load(bt_row + 2)
    end_m = tl.load(bt_row + 3)

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_h = pid_h * HEAD_DIM + tl.arange(0, HEAD_DIM)

    q_mask = offs_m < end_m
    
    offs_q = offs_m[:, None] * D_MODEL + offs_h[None, :]
    offs_k = (offs_n[None, :] + start_n) * D_MODEL + offs_h[:, None]
    offs_v = (offs_n[:, None] + start_n) * D_MODEL + offs_h[None, :]
    
    # Flash Elastic-Cache Attention
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q + offs_q, mask=q_mask[:, None], other=0.0)

    for n in range(0, end_n, BLOCK_N):
    # for n in range(0, SEQLEN_MAX, BLOCK_N):
        n = tl.multiple_of(n, BLOCK_N)
        kv_mask = (offs_n + n) < end_n
        k = tl.load(K + offs_k + n * D_MODEL, mask=kv_mask[None, :], other=0.0)
        v = tl.load(V + offs_v + n * D_MODEL, mask=kv_mask[:, None], other=0.0)

        qk = tl.dot(q, k)
        qk += tl.where(kv_mask[None, :], 0, float("-inf"))

        # attention tracking
        tl.store(
            S + (offs_m - start_m)[:, None] * stride_sm + pid_h * stride_sh + batch_id * stride_sb + (n + offs_n)[None, :], 
            qk, mask=(q_mask[:, None] & kv_mask[None, :]),
        )

        qk = qk * softmax_scale
        m_ij = tl.maximum(tl.max(qk, 1), m_i)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        acc_o_scale = tl.exp(m_i - m_ij)
        
        acc_o = acc_o * acc_o_scale[:, None]
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
        m_i = m_ij

    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    tl.store(O + offs_q, acc_o, mask=q_mask[:, None])


@triton.jit
def _flash_tracked_attention_fwd(
    Q, O, # (acc_q_len, d_model)
    K, V, # (acc_k_len, d_model)
    block_table,
    softmax_scale,
    NUM_BLOCKS: tl.constexpr,
    SEQLEN_MAX: tl.constexpr,
    D_MODEL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr,
):
    # step_m = tl.num_programs(0)
    pid_m = tl.program_id(0)  # block over query rows
    pid_h = tl.program_id(1)  # head

    # for m_idx in tl.range(pid_m, NUM_BLOCKS, step_m, num_stages=num_stages):

    bt_row = block_table + pid_m * 4
    batch_id = tl.load(bt_row + 0)
    start_n = batch_id * SEQLEN_MAX
    end_n = tl.load(bt_row + 1)
    start_m = tl.load(bt_row + 2)
    end_m = tl.load(bt_row + 3)

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_h = pid_h * HEAD_DIM + tl.arange(0, HEAD_DIM)

    q_mask = offs_m < end_m
    
    offs_q = offs_m[:, None] * D_MODEL + offs_h[None, :]
    offs_k = (offs_n[None, :] + start_n) * D_MODEL + offs_h[:, None]
    offs_v = (offs_n[:, None] + start_n) * D_MODEL + offs_h[None, :]
    
    # Flash Elastic-Cache Attention
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q + offs_q, mask=q_mask[:, None], other=0.0)

    for n in range(0, end_n, BLOCK_N):
    # for n in range(0, SEQLEN_MAX, BLOCK_N):
        n = tl.multiple_of(n, BLOCK_N)
        kv_mask = (offs_n + n) < end_n

        k = tl.load(K + offs_k + n * D_MODEL, mask=kv_mask[None, :], other=0.0)
        v = tl.load(V + offs_v + n * D_MODEL, mask=kv_mask[:, None], other=0.0)

        qk = tl.dot(q, k)
        qk += tl.where(kv_mask[None, :], 0, float("-inf"))
        
        qk = qk * softmax_scale
        m_ij = tl.maximum(tl.max(qk, 1), m_i)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        acc_o_scale = tl.exp(m_i - m_ij)
        
        acc_o = acc_o * acc_o_scale[:, None]
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
        m_i = m_ij

    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    tl.store(O + offs_q, acc_o, mask=q_mask[:, None])


@triton.jit
def _flash_verify_qkv_proj_fwd(
    Xn, Q1, K1, V1, # (acc_q_len, d_model)
    Pos,
    Wq, Wk, Wv, # (d_model, d_model)
    RotarySin, RotaryCos, # (acc_k_len, head_dim)
    block_table,
    HALF: tl.constexpr,
    D_MODEL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)  # block over query rows
    pid_h = tl.program_id(1)  # head

    bt_row = block_table + pid_m * 4
    start_n = tl.load(bt_row + 0)
    end_n = tl.load(bt_row + 1)
    start_m = tl.load(bt_row + 2)
    end_m = tl.load(bt_row + 3)
    
    range_half = tl.arange(0, HALF)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_h_half = pid_h * HEAD_DIM + range_half
    offs_d = tl.arange(0, BLOCK_D)

    q_mask = offs_m < end_m
    pos = tl.load(Pos + offs_m, mask=q_mask, other=0.0)
    
    offs_q_half = offs_m[:, None] * D_MODEL + offs_h_half[None, :]
    offs_x = offs_m[:, None] * D_MODEL + offs_d[None, :]

    offs_w1 = offs_h_half[None, :] * D_MODEL + offs_d[:, None]
    offs_w2 = (offs_h_half + HALF)[None, :] * D_MODEL + offs_d[:, None]

    # QKV projection for a single head
    acc_q1 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    acc_k1 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    acc_v1 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)

    acc_q2 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    acc_k2 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    acc_v2 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)

    for d in range(0, D_MODEL, BLOCK_D):
        x = tl.load(Xn + offs_x + d, mask=q_mask[:, None], other=0.0)

        wq1 = tl.load(Wq + offs_w1 + d)
        wk1 = tl.load(Wk + offs_w1 + d)
        wv1 = tl.load(Wv + offs_w1 + d)

        wq2 = tl.load(Wq + offs_w2 + d)
        wk2 = tl.load(Wk + offs_w2 + d)
        wv2 = tl.load(Wv + offs_w2 + d)

        acc_q1 += tl.dot(x, wq1)
        acc_k1 += tl.dot(x, wk1)
        acc_v1 += tl.dot(x, wv1)

        acc_q2 += tl.dot(x, wq2)
        acc_k2 += tl.dot(x, wk2)
        acc_v2 += tl.dot(x, wv2)

    # Apply rotary for a single head.
    offs_rot = pos[:, None] * HEAD_DIM + range_half[None, :]
    sin1 = tl.load(RotarySin + offs_rot, mask=q_mask[:, None], other=0.0)
    sin2 = tl.load(RotarySin + offs_rot + HALF, mask=q_mask[:, None], other=0.0)
    cos1 = tl.load(RotaryCos + offs_rot, mask=q_mask[:, None], other=0.0)
    cos2 = tl.load(RotaryCos + offs_rot + HALF, mask=q_mask[:, None], other=0.0)

    rq1 = acc_q1 * cos1 - acc_q2 * sin1
    rq2 = acc_q2 * cos2 + acc_q1 * sin2
    rk1 = acc_k1 * cos1 - acc_k2 * sin1
    rk2 = acc_k2 * cos2 + acc_k1 * sin2

    tl.store(Q1 + offs_q_half, rq1, mask=q_mask[:, None])
    tl.store(Q1 + offs_q_half + HALF, rq2, mask=q_mask[:, None])

    tl.store(K1 + offs_q_half, rk1, mask=q_mask[:, None])
    tl.store(K1 + offs_q_half + HALF, rk2, mask=q_mask[:, None])

    tl.store(V1 + offs_q_half, acc_v1, mask=q_mask[:, None])
    tl.store(V1 + offs_q_half + HALF, acc_v2, mask=q_mask[:, None])


@triton.jit
def _flash_verify_attention_fwd(
    Q1, K1, V1, O, # (acc_q_len, d_model)
    K, V, # (acc_k_len, d_model)
    Pos_k, Attn_Mask,
    block_table,
    softmax_scale,
    D_MODEL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # block over query rows
    pid_h = tl.program_id(1)  # head

    bt_row = block_table + pid_m * 4
    start_n = tl.load(bt_row + 0)
    end_n = tl.load(bt_row + 1)
    start_m = tl.load(bt_row + 2)
    end_m = tl.load(bt_row + 3)
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_h = pid_h * HEAD_DIM + tl.arange(0, HEAD_DIM)
    offs_q = offs_m[:, None] * D_MODEL + offs_h[None, :]

    q_mask = offs_m < end_m

    q = tl.load(Q1 + offs_q, mask=q_mask[:, None], other=0.0)
    k1 = tl.load(K1 + offs_q, mask=q_mask[:, None], other=0.0)
    v1 = tl.load(V1 + offs_q, mask=q_mask[:, None], other=0.0)

    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # KV Cache Attention
    for n in range(start_n, end_n, BLOCK_N):
        n = tl.multiple_of(n, BLOCK_N)
        current_n = offs_n + n
        kv_mask = current_n < end_n
        pos_k = tl.load(Pos_k + current_n, mask=kv_mask, other=0.0)
        offs_k = pos_k[None, :] * D_MODEL + offs_h[:, None]
        offs_v = pos_k[:, None] * D_MODEL + offs_h[None, :]

        k = tl.load(K + offs_k, mask=kv_mask[None, :], other=0.0)
        v = tl.load(V + offs_v, mask=kv_mask[:, None], other=0.0)

        qk = tl.dot(q, k)
        qk += tl.where(kv_mask[None, :], 0, float("-inf"))
        
        qk = qk * softmax_scale
        m_ij = tl.maximum(tl.max(qk, 1), m_i)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        acc_o_scale = tl.exp(m_i - m_ij)
        
        acc_o = acc_o * acc_o_scale[:, None]
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
        m_i = m_ij

    # Query block self-attention
    offs_mask = tl.arange(0, BLOCK_M)
    attn_mask = tl.load(Attn_Mask + (pid_m * BLOCK_M + offs_mask)[:, None] * BLOCK_M + offs_mask[None, :])
    # attn_mask = tl.load(Attn_Mask + offs_mask[:, None] * BLOCK_M + offs_mask[None, :])

    qk1 = tl.dot(q, tl.trans(k1)) * softmax_scale
    qk1 += tl.where(q_mask[None, :], 0, float("-inf"))
    qk1 += tl.where(attn_mask, 0, float("-inf"))
    
    p1 = tl.exp(qk1 - m_i[:, None])
    l_ij = tl.sum(p1, 1)
    
    p1 = p1.to(v1.dtype)
    acc_o += tl.dot(p1, v1)

    l_i_new = tl.exp(lse_i - m_i) + l_ij
    lse_i = m_i + tl.log(l_i_new)
    
    # Write to output
    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    tl.store(O + offs_q, acc_o, mask=q_mask[:, None])




def rotate_half(x: torch.Tensor) -> torch.Tensor:
    B, nh, T, hs = x.size()
    x = x.view(B, nh, T, 2, hs // 2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return ((t * pos_cos) + (rotate_half(t) * pos_sin)).to(t.dtype)

def rotary_emb(q: torch.Tensor, k: torch.Tensor, pos_sin, pos_cos):
    
    q_, k_ = q.float(), k.float()

    with torch.autocast(q.device.type, enabled=False):
        k_ = apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        q_ = apply_rotary_pos_emb(pos_sin, pos_cos, q_)     
        
        return q_.type_as(q), k_.type_as(k)
    

def make_blocks(j: int, start_m: int, end_m: int, block_m: int, device=None):
    starts = torch.arange(start_m, end_m, block_m, device=device)
    ends = starts + block_m
    ends[-1] = end_m
    batch = torch.full_like(starts, j)
    return torch.stack((batch, starts, ends), dim=1)


def flash_fused_elastic_cache(
    self, x: torch.Tensor, block_idx, positions, lengths,
    softmax_scale: float = None,
):
    """
    Project new hidden states h -> q/k/v, write into caches at per-token positions,
    then run FlashAttention for those query tokens against full KV cache.

    example batch size = 5, each squence in batch have different length.
    key_length: (487, 392, 443, 394, 432)
    query_length: (17, 18, 22, 17, 19)
    track_lenth: (1, 2, 6, 1, 3)

    assert: block_m >= track_length + query_mask_length

    """
    query_pos_flat, key_pos_flat, rotary_emb_pos, info, attn_scores, attn_mask = positions
    start_layer, query_verify_blocks, query_masked_blocks, query_tracked_blocks, query_blocks, active_batch, num_active, max_length, block_m, block_n, elastic_cache, verify = lengths
    
    # block_n = 128
    block_d = 64

    # apply_rotary = self.config.rope
    head_dim = self.config.d_model // self.config.n_heads
    d_model = self.config.d_model
    n_heads = self.config.n_heads
    half = head_dim // 2
    # hidden_size = self.hidden_size

    batch_size = len(active_batch)
    softmax_scale = softmax_scale or (1.0 / math.sqrt(head_dim))
    
    x_normed = self.attn_norm(x)
    acc_seqlen_q, d_model = x.shape
    att = torch.empty_like(x) 
    q = torch.empty_like(x)    

    if verify:
        k = torch.empty_like(x)  
        v = torch.empty_like(x)  

        num_warps = 4
        num_stages = 2
        grid = (query_verify_blocks.shape[0], n_heads)
        _flash_verify_qkv_proj_fwd[grid](  
            x_normed, q, k, v,
            query_pos_flat,
            self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
            rotary_emb_pos[0], rotary_emb_pos[1],
            query_verify_blocks,
            HALF=half,
            D_MODEL=d_model,
            HEAD_DIM=head_dim,
            BLOCK_M=block_m * 2,
            BLOCK_D=block_d // 2,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        num_warps = 4
        num_stages = 2
        grid = (query_verify_blocks.shape[0], n_heads)
        _flash_verify_attention_fwd[grid](  
            q, k, v, att,
            self.k_cache, self.v_cache, 
            key_pos_flat, attn_mask,
            query_verify_blocks, softmax_scale,
            D_MODEL=d_model,
            HEAD_DIM=head_dim,
            BLOCK_M=block_m * 2,
            BLOCK_N=block_n // 2,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
    
        S = torch.zeros((block_m, n_heads, batch_size, max_length), dtype=torch.float32, device=x.device) 
        stride_sm, stride_sh, stride_sb = S.stride(0), S.stride(1), S.stride(2)

        if query_blocks.shape[0] > 0:
            num_warps = 4
            num_stages = 2
            num_blocks = query_blocks.shape[0]

            grid = (num_blocks, n_heads)
            _flash_qkv_proj_cache_fwd[grid](  
                x_normed, q, self.k_cache, self.v_cache, query_pos_flat, query_blocks,
                self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
                rotary_emb_pos[0], rotary_emb_pos[1],
                D_MODEL=d_model,
                HALF=half,
                HEAD_DIM=head_dim,
                BLOCK_M=block_m,
                BLOCK_D=block_d,
                NUM_BLOCKS=num_blocks,
                num_warps=num_warps,
                num_stages=num_stages,
            ) 

        if query_masked_blocks.shape[0] > 0:
            num_warps = 4
            num_stages = 2
            grid = (query_masked_blocks.shape[0], n_heads)
            _flash_masked_attention_fwd[grid](  
                q, att, self.k_cache, self.v_cache, S,
                stride_sm, stride_sh, stride_sb,
                query_masked_blocks,
                softmax_scale,
                SEQLEN_MAX=max_length,
                D_MODEL=d_model,
                HEAD_DIM=head_dim,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=num_warps,
                num_stages=num_stages,
            )

            lengths[0] = start_layer
            positions[0] = query_pos_flat
            lengths[3] = query_tracked_blocks
            query_blocks = torch.cat([query_masked_blocks, query_tracked_blocks], dim=0)
            lengths[4] = query_blocks

            # print(block_idx, start_layer, acc_seqlen_q, query_tracked_blocks.shape, query_pos_flat.shape, query_blocks.shape, x.shape, att.shape)
            # print(query_tracked_blocks)


        if query_tracked_blocks.shape[0] > 0:
            num_warps = 4
            num_stages = 2
            num_blocks = query_tracked_blocks.shape[0]
            
            grid = (num_blocks, n_heads)
            _flash_tracked_attention_fwd[grid](  
                q, att, self.k_cache, self.v_cache,
                query_tracked_blocks,
                softmax_scale,
                NUM_BLOCKS=num_blocks,
                SEQLEN_MAX=max_length,
                D_MODEL=d_model,
                HEAD_DIM=head_dim,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        # if block_idx >= 7 and block_idx <= 22 :     
        #     attn_scores += S.mean((0, 1)).view(-1)


                
        if block_idx > 1:     
            attn_scores += S.mean((0, 1)).view(-1)


    att = self.attn_out(att)
    x = x + self.dropout(att)

    # Add feed-forward projection.
    # shape: (batch_size, seq_len, d_model)
    og_x = x
    if self._activation_checkpoint_fn is not None:
        x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
    else:
        x = self.ff_norm(x)
    x, x_up = self.ff_proj(x), self.up_proj(x) # new add
    if self._activation_checkpoint_fn is not None:
        x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
    else:
        x = self.act(x)
    x = x * x_up # new add
    x = self.ff_out(x)
    x = self.dropout(x)
    x = og_x + x

    return x.unsqueeze(0)






