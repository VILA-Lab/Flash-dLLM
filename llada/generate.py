# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM
# from elastic_cache_triton import get_rotary_embedding, _init_block_table, _update_block_table
import math
import triton
import triton.language as tl
import itertools

import heapq

import heapq

import heapq
from typing import List, Tuple, Any, Optional

import heapq
from typing import List, Tuple, Any, Optional
import time
from torch import einsum



def get_rotary_embedding(seq_len: int, dim, rope_theta, device: torch.device):
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    seq = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = einsum("i , j -> i j", seq, inv_freq)
    positions = torch.cat((freqs, freqs), dim=-1)
    pos_sin, pos_cos = positions.sin(), positions.cos()
    return pos_sin, pos_cos

def make_blocks(j: int, seqlen_k, max_length, start_m: int, end_m: int, block_m: int, device=None):
    q_starts = torch.arange(start_m, end_m, block_m, device=device)
    q_ends = q_starts + block_m
    q_ends[-1] = end_m
    batchs = torch.full_like(q_starts, j)
    k_ends = torch.full_like(q_starts, seqlen_k[j])
    return torch.stack((batchs, k_ends, q_starts, q_ends), dim=1)


@ torch.no_grad()
def generate_with_Flash_dLLM(model, prompts, prompt_lengths, batch_size, responses, n_steps, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=0.9, gamma=0.9, track_num=4, mask_num=4, eos_id=126081, verify=False,sliding_window=True, is_instruct=True, tokenizer=None, stop_tokens=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
    '''
    
    block_m = block_length
    block_n = 128
    max_length = max(prompt_lengths) + gen_length + block_m
    max_length = (max_length // block_n + 1) * block_n

    d_model = model.config.d_model
    n_layers = model.config.n_layers
    n_heads = model.config.n_heads
    num_tokens = 0
    max_num_tokens = 0
    num_nfe = 0
    info = []
    empty_int32 = torch.tensor([], device=model.device, dtype=torch.int32)
    
    count = 0
    x = torch.full((batch_size * max_length,), mask_id, dtype=torch.long, device=model.device)
    x_draft = torch.full((batch_size * max_length,), mask_id, dtype=torch.long, device=model.device)
    full_pos = torch.arange(batch_size * max_length, device=model.device).reshape(batch_size, max_length)
    attn_scores = torch.zeros((batch_size * max_length,), dtype=torch.float32, device=model.device)
    num_decoded = []
    num_newly_decoded = []
    query_masked_pos = []
    query_masked_blocks = []
    query_tracked_pos = []
    query_tracked_blocks = []
    query_verify_pos = []

    predicted_length = []
    decoded_eos_pos = []
    start_layer = []
    active_batch = []
    acc_seqlen_q = 0
    seqlen_keep = []
    verify_mode = []
    
    seqlen_k = []

    verify_gamma = torch.full((block_m,), gamma, device=model.device).cumprod_(0)
    print(mask_num, track_num, verify_gamma)

    # masked_m = []

    for j in range(batch_size):
        x[j * max_length : j * max_length + prompt_lengths[count]] = prompts[count]
        x_draft[j * max_length : j * max_length + prompt_lengths[count]] = prompts[count]
        x[j * max_length + prompt_lengths[count] + gen_length : (j + 1) * max_length] = eos_id
        seqlen_k.append((prompt_lengths[count] + gen_length) // block_n * block_n + block_n)
        num_decoded.append(prompt_lengths[count])
        num_newly_decoded.append(0)
        query_masked_pos.append(
            full_pos[j, num_decoded[j] : num_decoded[j] + block_m], 
        )

        query_tracked_pos.append(
            torch.cat([
                full_pos[j, : num_decoded[j]],
                full_pos[j, num_decoded[j] + block_m : seqlen_k[j]]
            ], dim=0)
        )

        predicted_length.append(num_decoded[j] + gen_length + j * max_length)
        decoded_eos_pos.append(-1)
        start_layer.append(-1)
        active_batch.append(count)
        query_verify_pos.append(empty_int32)
        seqlen_keep.append(0)
        verify_mode.append(0)
        count += 1
        

    for j, i in enumerate(active_batch):
        if query_masked_pos[j].shape[0] > 0:
            # query_masked_blocks.append((j, acc_seqlen_q, acc_seqlen_q + query_masked_pos[j].shape[0]))
            new_blocks = make_blocks(j, seqlen_k, max_length, acc_seqlen_q, acc_seqlen_q + query_masked_pos[j].shape[0], block_m, model.device)
            query_masked_blocks.append(new_blocks)
            acc_seqlen_q += query_masked_pos[j].shape[0]

    for j, i in enumerate(active_batch):
        if query_tracked_pos[j].shape[0] > 0:
            new_blocks = make_blocks(j, seqlen_k, max_length, acc_seqlen_q, acc_seqlen_q + query_tracked_pos[j].shape[0], block_m, model.device)
            query_tracked_blocks.append(new_blocks)
            # masked_m.append([acc_seqlen_q + num_decoded[j], acc_seqlen_q + num_decoded[j] + block_m * 2])
            acc_seqlen_q += query_tracked_pos[j].shape[0]

    # query_masked_blocks = torch.tensor(query_masked_blocks, device=model.device, dtype=torch.int32)
    query_masked_blocks = torch.cat(query_masked_blocks, dim=0)
    query_verify_blocks = empty_int32
    query_tracked_blocks = torch.cat(query_tracked_blocks, dim=0)
    num_verify = 0
    num_active = query_masked_blocks.shape[0]
    
    query_pos_flat = torch.cat(query_masked_pos + query_tracked_pos, dim=0)
    x_query = x[query_pos_flat].unsqueeze(0)
    key_pos_flat = empty_int32
    x_verify = []
    attn_mask = []

    pos_sin, pos_cos = get_rotary_embedding(max_length, model.config.d_model // model.config.n_heads, model.config.rope_theta, model.device)
    rotary_emb_pos = [pos_sin.repeat(batch_size, 1), pos_cos.repeat(batch_size, 1)]

    elastic_cache = None

    for l, block in enumerate(model.model.transformer.blocks):
        block.k_cache = torch.empty((max_length * batch_size, d_model), dtype=model.dtype, device=model.device)
        block.v_cache = torch.empty((max_length * batch_size, d_model), dtype=model.dtype, device=model.device)
        if elastic_cache is not None:
            block.x_cache = torch.empty((max_length * batch_size, d_model), dtype=model.dtype, device=model.device)
            block.q_cache = torch.empty((max_length * batch_size, d_model), dtype=model.dtype, device=model.device)

    print(f'Start decoding ..., Max length: {max_length}, max prefill: {max(prompt_lengths) + gen_length} generation length: {gen_length}, num samples: {len(prompts)}, batch size: {batch_size}, block M: {block_m}, verify: {verify}')
    if verify:
        print("Verification is enabled.")
    else:
        print("Verification is disabled.")

    # start_time = time.time()
    while True:        
        start_time = time.time()

        query_blocks = torch.cat([query_masked_blocks, query_tracked_blocks], dim=0)
        positions = [query_pos_flat, key_pos_flat, rotary_emb_pos, info, attn_scores, attn_mask]
        lengths = [start_layer, None, query_masked_blocks, query_tracked_blocks, query_blocks, active_batch, num_active, max_length, block_m, block_n, elastic_cache, False]
        output = model(x_query, use_cache=True, lengths=lengths, positions=positions)
        logits = output.logits.squeeze(0)    
        x_query = x_query.squeeze(0)
        
        acc_seqlen_masked = 0
        x_verify = []
        key_pos_flat = []
        attn_mask = []
        acc_seqlen_q = 0
        acc_seqlen_k = 0
        query_verify_blocks = []

        if verify:
            for j, i in enumerate(active_batch):
                if i == -1: continue

                # logits_masked_j = logits[acc_seqlen_masked : acc_seqlen_masked + query_masked_pos[j].shape[0]]
                logits_masked_j = logits[acc_seqlen_masked : acc_seqlen_masked + block_m]
                acc_seqlen_masked += query_masked_pos[j].shape[0]
                query_masked_pos[j] = query_masked_pos[j][:block_m]
                p_masked = F.softmax(logits_masked_j.to(torch.float64), dim=-1)
                x0_p_masked, x0_masked = torch.max(p_masked, dim=-1)

                if decoded_eos_pos[j] != -1:
                    length_mask = query_masked_pos[j] >= predicted_length[j]
                    x0_p_masked[length_mask] = 0
                    length_num = query_masked_pos[j].shape[0] - length_mask.sum()
                else:
                    length_num = query_masked_pos[j].shape[0]

                # x0_p_masked[block_m:] = 0

                sorted_val, sorted_idx = x0_p_masked.sort(descending=True)
                keep_idx = (x0_p_masked >= (threshold))
                keep_num = keep_idx.sum().item()
                keep_num = max(keep_num, 1)

                x_draft[query_masked_pos[j]] = x0_masked


                full_pos[j, num_decoded[j] : num_decoded[j] + query_masked_pos[j].shape[0]] = query_masked_pos[j][sorted_idx]
                pos_decoded_new_j = full_pos[j, num_decoded[j] : num_decoded[j] + keep_num]
                x0_decoded_new_j = x0_masked[sorted_idx][:keep_num]

                num_newly_decoded[j] = keep_num
                x[pos_decoded_new_j] = x0_decoded_new_j
                num_decoded[j] += keep_num

                if decoded_eos_pos[j] == -1:
                    pos_eos = pos_decoded_new_j[x0_decoded_new_j.eq(eos_id)]
                    if pos_eos.shape[0] > 0:
                        decoded_eos_pos[j] = pos_eos.max().item()
                        # predicted_length[j] = decoded_eos_pos[j] + 1 - j * max_length
                        predicted_length[j] = decoded_eos_pos[j] + 1

                
                num_verify = length_num - keep_num
                num_verify = max(num_verify, 0)
                num_verify = min(num_verify, block_m // 2)

                seqlen_keep[j] = block_m * 2 - num_verify * 2

                T = seqlen_keep[j]
                S = num_verify
                causal_mask = torch.ones((block_m * 2, block_m * 2), device=model.device, dtype=torch.bool)
                indices = torch.arange(num_verify, device=model.device)
                m11 = indices.view(-1, 1) >= indices.view(1, -1)
                m12 = indices.view(-1, 1) < indices.view(1, -1)
                m21 = indices.view(-1, 1) > indices.view(1, -1)
                m22 = indices.view(-1, 1) <= indices.view(1, -1)
                causal_mask[:T, T:T+S] = False
                causal_mask[T:T+S+S, T:T+S+S] = torch.cat([torch.cat([m11, m12], dim=1), torch.cat([m21, m22], dim=1)], dim=0)
                attn_mask.append(causal_mask)
                verify_mode[j] = 0
                
                query_verify_pos[j] = full_pos[j, num_decoded[j] : num_decoded[j] + num_verify]
                verify_tracked_pos_j = full_pos[j, num_decoded[j] - seqlen_keep[j] : num_decoded[j]]
                x_verify.append(torch.cat([x[verify_tracked_pos_j], x_draft[query_verify_pos[j]], x[query_verify_pos[j]]], dim=0))
                query_verify_pos[j] = torch.cat([verify_tracked_pos_j, query_verify_pos[j], query_verify_pos[j]], dim=0)

                key_pos_flat.append(
                    torch.cat([
                        full_pos[j, : num_decoded[j] - seqlen_keep[j]],
                        full_pos[j, num_decoded[j] + num_verify : seqlen_k[j]]
                    ], dim=0)
                )
                query_verify_blocks.append((
                    acc_seqlen_k, acc_seqlen_k + key_pos_flat[-1].shape[0],
                    acc_seqlen_q, acc_seqlen_q + query_verify_pos[j].shape[0],
                ))
                acc_seqlen_q += query_verify_pos[j].shape[0]
                acc_seqlen_k += key_pos_flat[-1].shape[0]


            query_verify_blocks = torch.tensor(query_verify_blocks, device=model.device, dtype=torch.int32)
            attn_mask = torch.cat(attn_mask, dim=0)

            query_pos_flat = torch.cat(query_verify_pos, dim=0)
            key_pos_flat = torch.cat(key_pos_flat, dim=0)
            x_query = torch.cat(x_verify, dim=0).unsqueeze(0)

            positions = [query_pos_flat, key_pos_flat, rotary_emb_pos, info, attn_scores, attn_mask]
            lengths = [start_layer, query_verify_blocks, None, query_tracked_blocks, None, active_batch, num_active, max_length, block_m, block_n, elastic_cache, True]
            output = model(x_query, use_cache=True, lengths=lengths, positions=positions)
            logits = output.logits.squeeze(0)    
            x_query = x_query.squeeze(0)

        acc_seqlen_verify = 0

        for j, i in enumerate(active_batch):
            if i == -1: continue
            n_steps[i] += 1

            # Get decoded tokens
            if verify:
                T = seqlen_keep[j]
                S = (block_m * 2 - T) // 2
                if S == 0:
                    keep_num = 0
                else:
                    logits_verify_j = logits[acc_seqlen_verify + T + S: acc_seqlen_verify + T + S + S]
                    x_verify_j = x_query[acc_seqlen_verify + T : acc_seqlen_verify + T + S]
                    p_verify = F.softmax(logits_verify_j.to(torch.float64), dim=-1)
                    x0_p_verify = p_verify.gather(1, x_verify_j.unsqueeze(1)).view(-1)
                    
                    if decoded_eos_pos[j] != -1:
                        x0_p_verify[query_pos_flat[acc_seqlen_verify + T : acc_seqlen_verify + T + S] >= predicted_length[j]] = 0

                    x0_p_verify = x0_p_verify.cumprod(dim=0)
                    keep_idx = (x0_p_verify >= gamma)
                    keep_num = keep_idx.sum().item()
                    
                    pos_decoded_new_j = full_pos[j, num_decoded[j] : num_decoded[j] + keep_num]
                    x0_decoded_new_j = x_verify_j[:keep_num]
                    
                acc_seqlen_verify += query_verify_pos[j].shape[0]
            else:
                logits_masked_j = logits[acc_seqlen_masked + num_verify : acc_seqlen_masked + num_verify + query_masked_pos[j].shape[0]]
                acc_seqlen_masked += query_masked_pos[j].shape[0]
                p_masked = F.softmax(logits_masked_j.to(torch.float64), dim=-1)
                x0_p_masked, x0_masked = torch.max(p_masked, dim=-1)

                if decoded_eos_pos[j] != -1:
                    x0_p_masked[query_masked_pos[j] >= predicted_length[j]] = 0

                sorted_val, sorted_idx = x0_p_masked.sort(descending=True)
                keep_idx = (sorted_val >= threshold)
                keep_num = keep_idx.sum().item()
                keep_num = max(keep_num, 1)

                full_pos[j, num_decoded[j] : num_decoded[j] + query_masked_pos[j].shape[0]] = query_masked_pos[j][sorted_idx]
                pos_decoded_new_j = full_pos[j, num_decoded[j] : num_decoded[j] + keep_num]
                x0_decoded_new_j = x0_masked[sorted_idx][:keep_num]
            
            if keep_num > 0:
                num_newly_decoded[j] += keep_num
                x[pos_decoded_new_j] = x0_decoded_new_j
                num_decoded[j] += keep_num

                if decoded_eos_pos[j] == -1:
                    pos_eos = pos_decoded_new_j[x0_decoded_new_j.eq(eos_id)]
                    if pos_eos.shape[0] > 0:
                        decoded_eos_pos[j] = pos_eos.max().item()
                        predicted_length[j] = decoded_eos_pos[j] + 1
            
            # Preparing for next iteration
            if (full_pos[j, num_decoded[j]:] < predicted_length[j]).any():
                decoded_ranks = attn_scores[full_pos[j, : num_decoded[j] - keep_num]].sort(dim=0, descending=False).indices
                full_pos[j, : num_decoded[j] - keep_num] = full_pos[j, : num_decoded[j] - keep_num][decoded_ranks]
                
                query_masked_pos[j] = full_pos[j, num_decoded[j] : num_decoded[j] + block_m]
                if track_num < 0:
                    query_tracked_pos[j] = torch.cat([
                        full_pos[j, : num_decoded[j]],
                        full_pos[j, num_decoded[j] + block_m : seqlen_k[j]]
                    ], dim=0)
                else:
                    track_start = max(0, num_decoded[j] - track_num * block_m)
                    # query_tracked_pos[j] = full_pos[j, track_start : num_decoded[j]]
                    track_end = min(seqlen_k[j], num_decoded[j] + block_m * mask_num)
                    query_tracked_pos[j] = torch.cat([
                        full_pos[j, track_start : num_decoded[j]],
                        full_pos[j, num_decoded[j] + block_m : track_end]
                    ], dim=0)

                start_layer[j] = n_layers

            else:
                ablation_length = True
                generated_answer_ids = x[j * max_length + prompt_lengths[i] : j * max_length + num_decoded[j]]
                                
                generated_answer_i = tokenizer.decode(generated_answer_ids, skip_special_tokens=False)
                for stop_seq in stop_tokens:
                    if stop_seq in generated_answer_i:
                        generated_answer_i = generated_answer_i.split(stop_seq)[0]
                generated_answer_ids = torch.tensor(tokenizer(generated_answer_i)["input_ids"])
                generated_answer_i = tokenizer.decode(generated_answer_ids, skip_special_tokens=True)

                num_tokens += (num_decoded[j] - prompt_lengths[i])
                max_num_tokens += gen_length
                responses[i] = generated_answer_i
                print('=' * 20)
                print(f'batch: {i} / {len(prompts)}, num steps: {n_steps[i]}, prefill length: {prompt_lengths[i]}, generated length: {num_decoded[j] - prompt_lengths[i]}, Tokens/iter: {(num_decoded[j] - prompt_lengths[i]) / n_steps[i]}, memory: {torch.cuda.max_memory_allocated() / 1e9} GB')
                print(generated_answer_i)
                print('=' * 20, end='\n\n')
                

                if count < len(prompts):
                    active_batch[j] = count
                    x[j * max_length : (j + 1) * max_length] = mask_id
                    x[j * max_length : j * max_length + prompt_lengths[count]] = prompts[count]
                    x[j * max_length + prompt_lengths[count] + gen_length : (j + 1) * max_length] = eos_id
                    seqlen_k[j] = (prompt_lengths[count] + gen_length) // block_n * block_n + block_n

                    x_draft[j * max_length : (j + 1) * max_length] = mask_id
                    x_draft[j * max_length : j * max_length + prompt_lengths[count]] = prompts[count]

                    num_decoded[j] = prompt_lengths[count]
                    num_newly_decoded[j] = 0
                    full_pos[j] = torch.arange(j * max_length, (j + 1) * max_length, device=model.device)
                    query_verify_pos[j] = empty_int32
                    query_masked_pos[j] = full_pos[j, num_decoded[j] : num_decoded[j] + block_m]
                    query_tracked_pos[j] = torch.cat([
                        full_pos[j, : num_decoded[j]],
                        full_pos[j, num_decoded[j] + block_m : seqlen_k[j]]
                    ], dim=0)
                    predicted_length[j] = num_decoded[j] + gen_length + j * max_length
                    start_layer[j] = -1
                    decoded_eos_pos[j] = -1
                    count += 1
                else:
                    active_batch[j] = -1
                    query_masked_pos[j] = empty_int32
                    query_tracked_pos[j] = empty_int32
                    query_verify_pos[j] = empty_int32


        if sum(active_batch) == -batch_size:
            break
        

        acc_seqlen_q = 0
        query_masked_blocks = []
        query_tracked_blocks = []

        for j, i in enumerate(active_batch):
            if query_masked_pos[j].shape[0] > 0:
                new_blocks = make_blocks(j, seqlen_k, max_length, acc_seqlen_q, acc_seqlen_q + query_masked_pos[j].shape[0], block_m, model.device)
                query_masked_blocks.append(new_blocks)
                acc_seqlen_q += query_masked_pos[j].shape[0]

        for j, i in enumerate(active_batch):
            if query_tracked_pos[j].shape[0] > 0:
                new_blocks = make_blocks(j, seqlen_k, max_length, acc_seqlen_q, acc_seqlen_q + query_tracked_pos[j].shape[0], block_m, model.device)
                query_tracked_blocks.append(new_blocks)
                acc_seqlen_q += query_tracked_pos[j].shape[0]

        if len(query_masked_blocks) > 0:
            query_masked_blocks = torch.cat(query_masked_blocks, dim=0)
        else:
            query_masked_blocks = empty_int32

        num_active = query_masked_blocks.shape[0]      

        if len(query_tracked_blocks) > 0:
            query_tracked_blocks = torch.cat(query_tracked_blocks, dim=0)
        else:
            query_tracked_blocks = empty_int32
        query_pos_flat = torch.cat(query_masked_pos + query_tracked_pos, dim=0)
        x_query = x[query_pos_flat].unsqueeze(0)

        attn_scores.fill_(0)

        
    return responses, num_tokens, max_num_tokens


