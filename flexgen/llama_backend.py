"""Implement tensor computations with pytorch."""
from enum import Enum, auto
from functools import partial
from itertools import count
import os
import queue
import shutil
import time
import threading
from typing import Optional, Union, Tuple
from flexgen.pytorch_backend import TorchDevice, TorchTensor, DeviceType
from flexgen import compression

import torch
import torch.nn.functional as F
import numpy as np
import math

from flexgen.utils import (GB, T, cpu_mem_stats, vector_gather,
    np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes)

general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None


def fix_recursive_import():
    global general_copy_compressed, TorchCompressedDevice, global_cpu_device
    from flexgen import compression
    general_copy_compressed = compression.general_copy_compressed
    TorchCompressedDevice = compression.TorchCompressedDevice


def rms_norm(input, weight, eps) -> torch.Tensor:
    input_dtype = input.dtype
    hidden_states = input.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


def rotary_embedding(x, inv_freq, seq_len):
    t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq.to(x.device))
    emb = torch.cat((freqs, freqs), dim=-1)
    return (
        emb.cos().to(x.dtype)[:seq_len].to(dtype=x.dtype),
        emb.sin().to(x.dtype)[:seq_len].to(dtype=x.dtype),
    )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=2):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


class LlamaTorchDevice(TorchDevice):

    def llama_input_embed(self, inputs, attention_mask, w_token, pad_token_id, donate):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        token_ids = inputs.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)

        return TorchTensor.create_from_torch(token_embed, self)

    def llama_output_embed(self, inputs, w_ln, w_token, eps, donate, do_sample, temperature):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)
        if donate[0]: inputs.delete()

        # output embedding
        # logits = F.linear(hidden, w_token.data)
        # last_token_logits = logits[:,-1,:]
        last_token_hidden = hidden[:, -1, :]
        last_token_logits = F.linear(last_token_hidden, w_token.data)

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)

    def llama_mha(self, inputs, position_ids, attention_mask, w_ln, w_q, w_k, w_v,
            w_re, w_out, n_head, n_kv_head, donate, eps, compress_cache, comp_config):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_re = w_re.device.decompress(w_re)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data) * scaling
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_kv_head, head_dim)
        v = v.view(b, s, n_kv_head, head_dim)

        kv_seq_len = k.shape[-3]
        cos, sin = rotary_embedding(v, w_re.data, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        n_kv_groups = n_head // n_kv_head
        k = repeat_kv(k, n_kv_groups)
        v = repeat_kv(v, n_kv_groups)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        return TorchTensor.create_from_torch(value, self), k, v

    def llama_mha_gen(self, inputs, position_ids, attention_mask, w_ln, w_q, w_k, w_v,
                w_re, w_out, eps, n_head, n_kv_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config):
        """Multi-head attention (decoding phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_re = w_re.device.decompress(w_re)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)

        # shape: (b, 1, h)
        q = F.linear(hidden, w_q.data) * scaling
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_kv_head, head_dim)
        v = v.view(b, tgt_s, n_kv_head, head_dim)

        cos, sin = rotary_embedding(v, w_re.data, seq_len=position_ids.max().item() + 1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        n_kv_groups = n_head // n_kv_head
        k = repeat_kv(k, n_kv_groups)
        v = repeat_kv(v, n_kv_groups)

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new

                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

                if k.is_cuda:
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim).cuda().half()
            else:  # Sparse attention
                # shape: (s, b * n_head, head_dim)
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)

        return TorchTensor.create_from_torch(value, self), k_new, v_new

    def llama_mlp(self, inputs, w_ln, w_g, w_u, w_d, eps, donate):
        # decompress weights
        if w_g.device.device_type == DeviceType.COMPRESSED:
            w_g = w_g.device.decompress(w_g)
            w_u = w_u.device.decompress(w_u)
            w_d = w_d.device.decompress(w_d)

        out = rms_norm(inputs.data, weight=w_ln.data, eps=eps)
        gate_out = F.linear(out, w_g.data)
        F.silu(gate_out, inplace=True)
        up_out = F.linear(out, w_u.data)
        out = F.linear(gate_out * up_out, w_d.data)
        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)

def rope_init_fn(config):
    base = config.rope_theta
    dim = config.head_dim
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    factor = config.rope_factor
    low_freq_factor = config.low_freq_factor
    high_freq_factor = config.high_freq_factor
    old_context_len = config.original_max_position_embeddings

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama

def llama3_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Llama3TorchDevice(TorchDevice):

    def __init__(self, name, mem_capacity=None, flops=None, rope_config=None):
        super().__init__(name, mem_capacity, flops)
        self.inv_freq = rope_init_fn(rope_config)

    def llama3_rotary_embedding(self, x, position_ids):
        inv_freq = self.inv_freq
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (
            emb.cos().to(dtype=x.dtype),
            emb.sin().to(dtype=x.dtype),
        )

    def llama_input_embed(self, inputs, attention_mask, w_token, pad_token_id, donate):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        token_ids = inputs.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)

        return TorchTensor.create_from_torch(token_embed, self)

    def llama_output_embed(self, inputs, w_ln, w_token, eps, donate, do_sample, temperature):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)
        if donate[0]: inputs.delete()

        # output embedding
        # logits = F.linear(hidden, w_token.data)
        # last_token_logits = logits[:,-1,:]
        last_token_hidden = hidden[:, -1, :]
        last_token_logits = F.linear(last_token_hidden, w_token.data)

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)
    
    def llama_mha_batched(self, inputs, position_ids, attention_mask, w_ln, w_q, w_k, w_v,
            w_out, n_head, n_kv_head, donate, eps, compress_cache, comp_config, batch_size=64):
        """Multi-head attention (prefill phase) with batched processing to reduce memory usage."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data) * scaling
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_kv_head, head_dim)
        v = v.view(b, s, n_kv_head, head_dim)

        cos, sin = self.llama3_rotary_embedding(v, position_ids)
        q, k = llama3_apply_rotary_pos_emb(q, k, cos, sin)

        n_kv_groups = n_head // n_kv_head
        k = repeat_kv(k, n_kv_groups)
        v = repeat_kv(v, n_kv_groups)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # Create output tensor
        value_out = torch.zeros((b, s, h), dtype=inputs.data.dtype, device=self.dev)
        
        # Process in batches to reduce memory usage
        num_batches = (s + batch_size - 1) // batch_size
        
        # Prepare causal mask once
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, s)
            current_len = end_idx - start_idx
            
            # Get the subset of q for this batch
            q_batch = q[:, start_idx:end_idx, :]  # (b * n_head, current_len, head_dim)
            
            # Calculate attention weights for this batch
            # shape: (b * n_head, current_len, s)
            attn_weights = torch.bmm(q_batch, k)
            
            # Apply attention mask
            # shape: (b, 1, current_len, s)
            batch_mask = attention_mask.data.view(b, 1, 1, s) & causal_mask[:, :, start_idx:end_idx, :]
            
            # shape: (b, n_head, current_len, s)
            attn_weights = attn_weights.view(b, n_head, current_len, s)
            attn_weights = torch.where(batch_mask, attn_weights, -1e4)
            attn_weights = attn_weights.view(b * n_head, current_len, s)
            attn_weights = F.softmax(attn_weights, dim=2)
            
            # Calculate output values for this batch
            # shape: (b, n_head, current_len, head_dim)
            value_batch = torch.bmm(attn_weights, v).view(b, n_head, current_len, head_dim)
            
            # shape: (b, current_len, h)
            value_batch = value_batch.transpose(1, 2).reshape(b, current_len, h)
            value_batch = F.linear(value_batch, w_out.data)
            
            # Add to output
            value_out[:, start_idx:end_idx, :] = value_batch + inputs.data[:, start_idx:end_idx, :]
            
            # Free memory
            del attn_weights, value_batch
            torch.cuda.empty_cache()

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        return TorchTensor.create_from_torch(value_out, self), k, v

    def llama_mha(self, inputs, position_ids, attention_mask, w_ln, w_q, w_k, w_v,
            w_out, n_head, n_kv_head, donate, eps, compress_cache, comp_config):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data) * scaling
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_kv_head, head_dim)
        v = v.view(b, s, n_kv_head, head_dim)

        cos, sin = self.llama3_rotary_embedding(v, position_ids)
        q, k = llama3_apply_rotary_pos_emb(q, k, cos, sin)

        n_kv_groups = n_head // n_kv_head
        k = repeat_kv(k, n_kv_groups)
        v = repeat_kv(v, n_kv_groups)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        return TorchTensor.create_from_torch(value, self), k, v

    def llama_mha_gen(self, inputs, position_ids, attention_mask, w_ln, w_q, w_k, w_v,
                w_out, eps, n_head, n_kv_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config):
        """Multi-head attention (decoding phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)

        # shape: (b, 1, h)
        q = F.linear(hidden, w_q.data) * scaling
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_kv_head, head_dim)
        v = v.view(b, tgt_s, n_kv_head, head_dim)

        cos, sin = self.llama3_rotary_embedding(v, position_ids)
        q, k = llama3_apply_rotary_pos_emb(q, k, cos, sin)

        n_kv_groups = n_head // n_kv_head
        k = repeat_kv(k, n_kv_groups)
        v = repeat_kv(v, n_kv_groups)

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new

                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

                if k.is_cuda:
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim).cuda().half()
            else:  # Sparse attention
                # shape: (s, b * n_head, head_dim)
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)

        return TorchTensor.create_from_torch(value, self), k_new, v_new

    def llama_mlp(self, inputs, w_ln, w_g, w_u, w_d, eps, donate):
        # decompress weights
        if w_g.device.device_type == DeviceType.COMPRESSED:
            w_g = w_g.device.decompress(w_g)
            w_u = w_u.device.decompress(w_u)
            w_d = w_d.device.decompress(w_d)

        out = rms_norm(inputs.data, weight=w_ln.data, eps=eps)
        gate_out = F.linear(out, w_g.data)
        F.silu(gate_out, inplace=True)
        up_out = F.linear(out, w_u.data)
        out = F.linear(gate_out * up_out, w_d.data)
        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)
