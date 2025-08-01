# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# from verl import DataProto
# from verl.trainer.ppo import core_algos
# from verl.trainer.ppo.core_algos import agg_loss
# from verl.workers.actor import BasePPOActor
# from verl.utils.py_functional import append_to_dict
# from verl.utils.torch_functional import logprobs_from_logits, log_probs_from_logits_all_rmpad
# from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
# import verl.utils.torch_functional as verl_F
from codetiming import Timer
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
from typing import Dict, Union, List, Optional

import os
import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

__all__ = ['RobDataParallelPPOActor']

def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        labels = labels.reshape(-1)
        output = logprobs_from_logits_flash_attn(logits, labels)
        output = output.view(*batch_dim)
    else:
        output = logprobs_from_logits_naive(logits, labels)
    return output


def logprobs_from_logits_flash_attn(logits, labels):
    output = -cross_entropy_loss(logits, labels)[0]
    return output


def logprobs_from_logits_naive(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logpy = gather_from_labels(logp, labels)
    return logpy


def logprobs_of_labels_v2(logits: torch.FloatTensor, labels):
    """
    A memory efficient implementation of logprobs_from_logits
    """
    assert logits.dtype == torch.float32, 'Using bf16 logits with logprobs_of_labels_v2 may lead to divergence'
    logprobs_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1))
    logprobs_labels = logprobs_labels - torch.logsumexp(logits, dim=-1, keepdim=True)
    return logprobs_labels.squeeze(-1)


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def masked_sum(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if (mask == False).all():
        return (values * mask).sum(axis=axis) 
    else: 
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)

def masked_mean_weighted(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor = None, axis=None):
    """
    带重要性权重的masked mean函数
    
    Args:
        values: 输入张量 (bs, seq_len)
        mask: 掩码张量 (bs, seq_len)
        weights: 重要性权重 (bs, seq_len) 或 (bs,) 或 None
    
    Returns:
        加权平均值
    """
    if weights is None:
        # 如果没有权重，回退到普通的masked mean
        return masked_mean(values, mask, axis=axis)
    
    # 确保权重维度正确
    if weights.dim() == 1 and values.dim() == 2:
        # weights: (bs,) -> (bs, seq_len)
        weights = weights.unsqueeze(1).expand_as(values)
    # import ipdb;ipdb.set_trace()
    # 计算加权和
    weighted_sum = torch.sum(values * mask * weights)
    
    # 计算有效权重总和
    weight_sum = torch.sum(mask * weights)
    
    if weight_sum > 0:
        return weighted_sum / weight_sum * weights[0][0]
    else:
        return torch.tensor(0.0, device=values.device, dtype=values.dtype)

def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def get_eos_mask(response_id: torch.Tensor, eos_token: int = 2, dtype=torch.int64):
    '''
    e.g. end of sentence token=1
    response_id: [0, 0, 2, 42, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    '''
    eos_mask = response_id.eq(eos_token).long()
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    return eos_mask


def compute_grad_norm(model: nn.Module):
    total_grad_square = 0
    total_params = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_square += torch.sum(torch.square(param.grad.detach())).item()
    return total_grad_square


def broadcast_dict_tensor(tensors: Union[Dict[str, torch.Tensor], TensorDict], src, group):
    """
    TODO: optimize this. Technically, we only need one broadcast
    """

    for key in tensors.sorted_keys:
        torch.distributed.broadcast(tensors[key], src=src, group=group, async_op=False)


def allgather_dict_tensors(tensors: Union[Dict[str, torch.Tensor], TensorDict], size, group, dim=0):
    """
    TODO: optimize this.
    - We can use async ops
    - We can use only one allgather
    Args:
        tensors:
        size:
        group:

    Returns:

    """
    if isinstance(tensors, TensorDict):
        is_tensor_dict = True
        tensors_as_dict = tensors.to_dict()
    else:
        tensors_as_dict = tensors
        is_tensor_dict = False

    output = {}
    sorted_keys = sorted(tensors_as_dict.keys())
    for key in sorted_keys:
        val = tensors_as_dict[key]
        output[key] = [torch.empty_like(val) for _ in range(size)]
        torch.distributed.all_gather(output[key], val, group=group, async_op=False)
        output[key] = torch.cat(output[key], dim=dim)

    if is_tensor_dict:
        output = TensorDict(source=output, batch_size=tensors.batch_size[0] * size)

    return output


def split_dict_tensor_into_batches(tensors: TensorDict, batch_size) -> List[TensorDict]:
    assert tensors.batch_size[0] % batch_size == 0, \
        f'input data batch size: {tensors.batch_size[0]}, split batch size: {batch_size}'
    return tensors.split(batch_size)


def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, 'constant', pad_token_id)


from transformers import PreTrainedTokenizer


def tokenize_and_postprocess_data(prompt: str,
                                  tokenizer: PreTrainedTokenizer,
                                  max_length: int,
                                  pad_token_id: int,
                                  left_pad=True,
                                  truncation='error'):
    """
    input_data is the output from tokenizer.
    """
    assert truncation in ['left', 'right', 'error']

    input_data = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)

    input_ids = input_data['input_ids']
    attention_mask = input_data['attention_mask']

    assert input_ids.ndim == 2

    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = pad_sequence_to_length(input_ids,
                                           max_seq_len=max_length,
                                           pad_token_id=pad_token_id,
                                           left_pad=left_pad)
        attention_mask = pad_sequence_to_length(attention_mask,
                                                max_seq_len=max_length,
                                                pad_token_id=0,
                                                left_pad=left_pad)
    elif sequence_length > max_length:
        if truncation == 'left':
            # actually, left truncation may not be reasonable
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == 'right':
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        elif truncation == 'error':
            raise NotImplementedError(f'{sequence_length} is larger than {max_length}')
        else:
            raise NotImplementedError(f'Unknown truncation method {truncation}')

    return input_ids, attention_mask


def postprocess_rob_data(input_ids,
                        attention_mask,
                        max_length: int,
                        pad_token_id: int,
                        left_pad=True,
                        truncation='error'):
    """
    input_data is the output from tokenizer.
    """
    assert truncation in ['left', 'right', 'error']

    assert input_ids.ndim == 2

    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = pad_sequence_to_length(input_ids,
                                           max_seq_len=max_length,
                                           pad_token_id=pad_token_id,
                                           left_pad=left_pad)
        attention_mask = pad_sequence_to_length(attention_mask,
                                                max_seq_len=max_length,
                                                pad_token_id=0,
                                                left_pad=left_pad)
    elif sequence_length > max_length:
        if truncation == 'left':
            # actually, left truncation may not be reasonable
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == 'right':
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        elif truncation == 'error':
            raise NotImplementedError(f'{sequence_length} is larger than {max_length}')
        else:
            raise NotImplementedError(f'Unknown truncation method {truncation}')

    return input_ids, attention_mask


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """ Remove the pad token. 

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[List[int]]): contains the rmpad token ids per query.
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        no_padding_batch.append((ids[len(ids) - mask.sum():]).cpu().numpy().tolist())
    return no_padding_batch


def log_probs_from_logits_response(input_ids, logits, response_length):
    """Compute the response log_probs from full logits. Note that logits = model(input_ids)
    
    Args:
        input_ids: [batch_size, seqlen]
        logits: [batch_size, seqlen, vocab_size]
    
    Returns:
        response_log_prob: 
    """
    response_logits = logits[:, -response_length - 1:-1]
    response = input_ids[:, -response_length:]
    response_log_prob = logprobs_from_logits(logits=response_logits, labels=response)
    return response_log_prob


def log_probs_from_logits_response_rmpad(input_ids, attention_mask, logits_rmpad, response_length):
    """Compute the log_probs from logits with rmpad logits and pad input. Note that
    logits_rmpad = model(input_ids_rmpad). For each sentences, there is a shift between
    logits and input_ids.
    The reason for this function to is to compute logprobs_from_logits in rmpad mode because it is memory-intensive
    for large vocab_size
    
    Args:
        input_ids: [batch_size, seqlen]
        attention_mask: [batch_size, seqlen]
        logits_rmpad: [total_nnz, vocab_size]
        response_length: int
    """
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size, seqlen = input_ids.shape
    input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(input_ids.unsqueeze(-1),
                                                                            attention_mask=attention_mask)
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)  # (total_nnz,)
    full_output = pad_input(hidden_states=full_log_probs_rmpad.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen)
    output = full_output.squeeze(-1)[:, -response_length - 1:-1]  # [batch_size, response_length]
    return output


def log_probs_from_logits_all_rmpad(input_ids_rmpad, logits_rmpad, indices, batch_size, seqlen, response_length):
    """Compute the log_probs from logits with rmpad input_ids and logits. Note that
    logits_rmpad = model(input_ids_rmpad). For each sentences, there is a shift between
    logits and input_ids.
    The reason for this function to is to compute logprobs_from_logits in rmpad mode because it is memory-intensive
    for large vocab_size
    
    Args:
        input_ids_rmpad: [1, total_nnz]
        logits_rmpad: [total_nnz, vocab_size]
        indices: [total_nnz]
        batch_size: int
        seqlen: int
        response_length: int
    """
    from flash_attn.bert_padding import pad_input
    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # transpose back to [total_nnz, 1]
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)  # (total_nnz,)
    full_output = pad_input(hidden_states=full_log_probs_rmpad.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen)
    output = full_output.squeeze(-1)[:, -response_length - 1:-1]  # [batch_size, response_length]
    return output


from transformers.generation.logits_process import (TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)


def post_process_logits(input_ids, logits, temperature, top_k, top_p):
    if temperature != 1.:
        logits = logits.div_(temperature)  # inplace operation to avoid OOM
    # TODO: add them back
    # if top_k is not None and top_k > 0:
    #     logits = TopKLogitsWarper(top_k=top_k)(input_ids, logits)
    # if top_p is not None and top_p < 1.0 and top_p > 0.0:
    #     logits = TopPLogitsWarper(top_p=top_p)(input_ids, logits)
    return logits


"""
Optimizer related
"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum lr ratio w.r.t the maximum.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.
    coef = (1 - min_lr_ratio) * 0.5
    intercept = (1 + min_lr_ratio) * 0.5

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        return max(0.0, x * coef + intercept)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
):

    def lr_lambda(current_step):
        return min(1, float(current_step) / float(max(1, num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype,
                                          tgt_len=input_shape[-1]).to(inputs_embeds.device)
        combined_attention_mask = (expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask +
                                   combined_attention_mask)

    return combined_attention_mask


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    # TODO: check if this is correct
    if loss_agg_mode == "token-mean":
        loss = masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss

def agg_loss_weighted(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str, 
                     importance_weights: torch.Tensor = None):
    """
    带重要性权重的损失聚合函数（基于原始agg_loss修改）
    
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
        importance_weights: `(torch.Tensor)` 
            shape: (bs, response_length) or (bs,) or None
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss with importance weights
    """
    if importance_weights is None:
        # 如果没有重要性权重，回退到原始实现
        return agg_loss(loss_mat, loss_mask, loss_agg_mode)
    
    # 确保重要性权重维度正确
    if importance_weights.dim() == 1 and loss_mat.dim() == 2:
        # importance_weights: (bs,) -> (bs, response_length)
        importance_weights = importance_weights.unsqueeze(1).expand_as(loss_mat)
    
    if loss_agg_mode == "token-mean":
        # 使用带权重的masked mean
        loss = masked_mean_weighted(loss_mat, loss_mask, importance_weights)
        
    elif loss_agg_mode == "seq-mean-token-sum":
        # 对每个序列：加权token-sum，然后取序列平均
        weighted_loss_mat = loss_mat * loss_mask * importance_weights
        seq_losses = torch.sum(weighted_loss_mat, dim=-1)  # weighted token-sum
        
        # 计算每个序列的权重总和用于归一化
        seq_weights = torch.sum(loss_mask * importance_weights, dim=-1)
        
        # 避免除零
        valid_seqs = seq_weights > 0
        if valid_seqs.any():
            # 归一化每个序列的损失
            normalized_seq_losses = torch.zeros_like(seq_losses)
            normalized_seq_losses[valid_seqs] = seq_losses[valid_seqs] / seq_weights[valid_seqs] * torch.sum(loss_mask[valid_seqs], dim=-1)
            loss = torch.mean(normalized_seq_losses[valid_seqs])  # seq-mean
        else:
            loss = torch.tensor(0.0, device=loss_mat.device, dtype=loss_mat.dtype)
            
    elif loss_agg_mode == "seq-mean-token-mean":
        # 对每个序列：加权token-mean，然后取序列平均
        seq_losses = []
        for i in range(loss_mat.size(0)):
            seq_loss_mat = loss_mat[i:i+1]
            seq_mask = loss_mask[i:i+1]
            seq_weights = importance_weights[i:i+1]
            
            # 计算单个序列的加权平均
            seq_loss = masked_mean_weighted(seq_loss_mat, seq_mask, seq_weights)
            seq_losses.append(seq_loss)
        
        # 对所有序列取平均
        loss = torch.mean(torch.stack(seq_losses))
        
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss

def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, clip_ratio_high, clip_ratio_low):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)

    pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl

def compute_policy_loss_weighted(old_log_prob, log_prob, advantages, eos_mask, 
                                clip_ratio_high, clip_ratio_low, importance_weights=None):
    """
    带重要性权重的PPO policy loss计算（基于原始compute_policy_loss修改）
    
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_high: (float)
            高clip比率
        clip_ratio_low: (float)
            低clip比率
        importance_weights: `(torch.Tensor)`
            shape: (bs, response_length) or (bs,) or None
            重要性采样权重

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO with importance weights
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            approximate KL divergence
    """
    if importance_weights is None:
        # 如果没有重要性权重，回退到原始实现
        return compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, 
                                 clip_ratio_high, clip_ratio_low)
    
    # 确保重要性权重维度正确
    if importance_weights.dim() == 1 and old_log_prob.dim() == 2:
        # importance_weights: (bs,) -> (bs, response_length)
        importance_weights = importance_weights.unsqueeze(1).expand_as(old_log_prob)
    
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    
    # 使用带权重的KL计算
    ppo_kl = masked_mean_weighted(-negative_approx_kl, eos_mask, importance_weights)

    # 计算policy loss
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)

    # 使用带权重的masked mean
    import ipdb;ipdb.set_trace()
    pg_loss = masked_mean_weighted(torch.max(pg_losses, pg_losses2), eos_mask, importance_weights)
    pg_clipfrac = masked_mean_weighted(torch.gt(pg_losses2, pg_losses).float(), eos_mask, importance_weights)
    
    return pg_loss, pg_clipfrac, ppo_kl

# modified: get the top top_ratio of entropy mask
def get_top_entropy_mask_simple(entropy, response_mask, top_ratio=0.2):
    """
    Get the top top_ratio of entropy mask
    Return:
        mask: (bsz, response_length)
    """
    with torch.no_grad():
        # get the valid entropy
        valid_entropy = entropy[response_mask.bool()]
        
        # calculate the threshold
        if len(valid_entropy) == 0:
            return torch.zeros_like(entropy, dtype=torch.bool), torch.zeros_like(entropy, dtype=torch.bool)
        
        k = max(1, int(len(valid_entropy) * top_ratio))
        threshold = torch.topk(valid_entropy, k=k, largest=True)[0][-1]
        
        # create the mask: entropy >= threshold and is valid position
        mask_entropy_greater_than_threshold = (entropy >= threshold) & response_mask.bool()
        mask_entropy_less_than_threshold = (entropy < threshold) & response_mask.bool()
        
        return mask_entropy_greater_than_threshold, mask_entropy_less_than_threshold

def compute_token_on_off_policy_loss(
    old_log_prob, 
    log_prob, 
    advantages, 
    eos_mask, 
    cliprange, 
    clip_upper_bound,
    prefix_mask, 
    off_cliprange, 
    off_normalize=False, 
    off_abs_cliprange=None, 
    off_max_clip=None, 
    off_min_clip=None,
    all_max_clip=None, 
    off_policy_reshape="no_reshape", 
    off_policy_reshape_weight=1.0, 
    off_policy_reshape_pow_exp=0.5,
    on_policy_reshape="no_reshape", 
    on_policy_reshape_weight=1.0,
    on_policy_reshape_pow_exp=0.5,
    target_probs=None,
    loss_remove_token_mean=False,
    loss_remove_clip=False,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        prefix_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    # off-policy loss
    # compute off-policy probability
    # import ipdb;ipdb.set_trace()
    if not prefix_mask.item():
        negative_approx_kl = log_prob - old_log_prob
        ppo_kl = masked_mean(-negative_approx_kl, eos_mask)

        if on_policy_reshape == "no_reshape":
            ratio = torch.exp(negative_approx_kl) # [bsz, l]
        elif on_policy_reshape == "logp":
            ratio = log_prob - old_log_prob
        elif on_policy_reshape == "p_logp":
            ratio = torch.exp(negative_approx_kl) + on_policy_reshape_weight * negative_approx_kl
        elif on_policy_reshape == "square_root":
            ratio = torch.exp(negative_approx_kl) # [bsz, l]
            ratio = torch.sqrt(ratio)
        elif on_policy_reshape == "pow":
            ratio = torch.exp(negative_approx_kl) # [bsz, l]
            ratio = torch.pow(ratio, on_policy_reshape_pow_exp)
        elif on_policy_reshape == "p_div_p_0.1":
            prob = torch.exp(log_prob)
            old_prob = torch.exp(old_log_prob)
            f_prob = prob / (prob + 0.1)
            f_old_prob = old_prob / (old_prob + 0.1)
            ratio = f_prob / f_old_prob
        elif on_policy_reshape == "p_div_p_0.5":
            prob = torch.exp(log_prob)
            old_prob = torch.exp(old_log_prob)
            f_prob = prob / (prob + 0.5)
            f_old_prob = old_prob / (old_prob + 0.5)
            ratio = f_prob / f_old_prob
        else:
            raise ValueError(f"Invalid on_policy_reshape: {on_policy_reshape}")

        on_pg_losses = -advantages * ratio
        upper_bound = max(clip_upper_bound, 1.0 + cliprange)
        if upper_bound == clip_upper_bound:
            print('clip upper bound is used: ', clip_upper_bound)

        if loss_remove_clip is False:
            on_pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, upper_bound)
            on_pg_clipfrac = masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)
            on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
            on_pg_loss = masked_mean(on_pg_losses, eos_mask)
        else:
            on_pg_loss = masked_mean(on_pg_losses, eos_mask)
            on_pg_clipfrac = torch.tensor(0.0)
        
        pg_losses = on_pg_loss
        off_pg_loss = torch.tensor(0.0)
        off_pg_clipfrac = torch.tensor(0.0)
        off_ratio_mean = torch.tensor(0.0)
        off_ratio_max_clip_frac = torch.tensor(0.0)
        off_ratio_min_clip_frac = torch.tensor(0.0)
        
        on_policy_probs = torch.exp(old_log_prob)
        on_policy_prob = masked_mean(on_policy_probs, eos_mask)
        if on_policy_prob.isnan().item() is True:
            on_policy_prob = torch.tensor(0.0)
        off_policy_prob = torch.tensor(0.0)
        
    else:
        # compute off-policy loss
        if target_probs is None:
            off_ratio = torch.exp(log_prob) # [bsz, l]
            if off_policy_reshape == "no_reshape":
                pass
            elif off_policy_reshape == "logp":
                off_ratio = log_prob * off_policy_reshape_weight
            elif off_policy_reshape == "p_logp":
                off_ratio = log_prob * off_policy_reshape_weight + off_ratio
            elif off_policy_reshape == "square_root":
                off_ratio = torch.sqrt(off_ratio)
            elif off_policy_reshape == "p_div_p_0.1":
                off_ratio = off_ratio / (off_ratio + 0.1)
            elif off_policy_reshape == "p_div_p_0.5":
                off_ratio = off_ratio / (off_ratio + 0.5)
            elif off_policy_reshape == "p_div_p_0.3":
                off_ratio = off_ratio / (off_ratio + 0.3)
            elif off_policy_reshape == "pow":
                off_ratio = torch.pow(off_ratio, off_policy_reshape_pow_exp)
            else:
                raise ValueError(f"Invalid off_policy_reshape: {off_policy_reshape}")
        else:
            assert target_probs.shape == log_prob.shape
            off_ratio = torch.exp(log_prob) / (target_probs+1e-6)
            # off_ratio[log_prob == 0] = 0
            # off_ratio = off_ratio * prefix_mask
            # assert ((target_probs > 0) == prefix_mask).all()
            
        # clip off-policy ratio
        if off_max_clip is not None:
            off_ratio = torch.clamp(off_ratio, max=off_max_clip)
            off_ratio_max_clip_frac = masked_mean((off_ratio == off_max_clip).float(), eos_mask)
        else:
            off_ratio_max_clip_frac = torch.tensor(0.0)
            
        if off_min_clip is not None:
            off_ratio = torch.clamp(off_ratio, min=off_min_clip)
            off_ratio_min_clip_frac = masked_mean((off_ratio == off_min_clip).float(), eos_mask)
        else:
            off_ratio_min_clip_frac = torch.tensor(0.0)

        off_ratio_mean = masked_mean(off_ratio, eos_mask)
        if off_ratio_mean.isnan().any().item():
            off_ratio_mean = torch.tensor(0.0)

        off_pg_losses = -advantages * off_ratio
        off_pg_loss = masked_mean(off_pg_losses, eos_mask)
        if off_pg_loss.isnan().item() is True:
            off_pg_loss = torch.tensor(0.0)
        off_pg_clipfrac = torch.tensor(0.0)
        
        pg_losses = off_pg_loss
        on_pg_loss = torch.tensor(0.0)
        on_pg_clipfrac = torch.tensor(0.0)
        ppo_kl = torch.tensor(0.0)
        
        off_policy_probs = torch.exp(log_prob)
        off_policy_prob = masked_mean(off_policy_probs, eos_mask)
        if off_policy_prob.isnan().item() is True:
            off_policy_prob = torch.tensor(0.0)
        on_policy_prob = torch.tensor(0.0)
    
    # prefix_mask = prefix_mask.float()
    # pg_losses = off_pg_losses * prefix_mask + on_pg_losses * (1 - prefix_mask)
    
    # log on/off probs
    
    
    # import ipdb;ipdb.set_trace()   
    # if all_max_clip is not None:
    #     p_on = torch.exp(log_prob)
    #     p_on_mask = (p_on <= all_max_clip).float()
    #     eos_mask = eos_mask * p_on_mask
    #     pg_losses = pg_losses * p_on_mask
        
    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
        print(f'no token mean: mean normalization {eos_mask.shape[-1]}')
    else:
        pg_loss = masked_mean(pg_losses, eos_mask)

    return {
        "pg_loss": pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_loss": on_pg_loss,
        "off_pg_clipfrac": off_pg_clipfrac,
        "on_pg_clipfrac": on_pg_clipfrac,
        "ppo_kl": ppo_kl,
        "off_policy_prob": off_policy_prob,
        "on_policy_prob": on_policy_prob,
        "off_ratio_mean": off_ratio_mean,
        "off_ratio_max_clip_frac": off_ratio_max_clip_frac,
        "off_ratio_min_clip_frac": off_ratio_min_clip_frac,
    }

# data_dict = {
#     'test_idx': test_idx,
#     'slice_id': slice_id,
#     'entropy': entropy,
#     'log_prob': log_prob,
#     'old_log_prob_tmp': old_log_prob_tmp,
#     'advantages_tmp': advantages_tmp,
#     'response_mask_tmp': response_mask_tmp,
# }

for xx in range(8):
    xx = 1
    tensor_dict = torch.load(f'/share/home/u16023/jinyiyang/experiments/SimpleVLA-RL/outputs/tensors_b0_t{xx}_s0_wmask.pt')
    loss_agg_mode = "token-mean"
    i = tensor_dict["i"]
    i_1 = tensor_dict["i+1"]
    traj_len = tensor_dict["traj_len"]
    traj_split_num = tensor_dict["traj_split_num"]
    entropy = tensor_dict["entropy"]
    log_prob = tensor_dict["log_prob"]
    old_log_prob_tmp = tensor_dict["old_log_prob_tmp"]
    advantages_tmp = tensor_dict["advantages_tmp"]
    response_mask_tmp = tensor_dict["response_mask_tmp"]
    input_ids = tensor_dict["input_ids"]
    input_ids_train = tensor_dict["input_ids_train"]
    # importance_weights_tmp = tensor_dict["importance_weights_tmp"]
    response_mask_sum = tensor_dict["response_mask_sum"]
    response_mask = tensor_dict["response_mask"]
    on_policy_mask = tensor_dict["on_policy_mask"]
    off_policy_mask = tensor_dict["off_policy_mask"]
    
    # import ipdb;ipdb.set_trace()

    cov = (advantages_tmp - advantages_tmp[response_mask_tmp].mean()) * (log_prob - log_prob[response_mask_tmp].mean()) * response_mask_tmp
    # ratio
    negative_approx_kl = log_prob - old_log_prob_tmp
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    if entropy is not None:
        response_mask_tmp, mask_entropy_less_than_threshold = get_top_entropy_mask_simple(entropy, response_mask_tmp, top_ratio=0.2)
        # lof metrics with different mask
        advantages_high_entropy = agg_loss(loss_mat=advantages_tmp, loss_mask=response_mask_tmp, loss_agg_mode=loss_agg_mode)
        advantages_low_entropy = agg_loss(loss_mat=advantages_tmp, loss_mask=mask_entropy_less_than_threshold, loss_agg_mode=loss_agg_mode)
        entropy_high = agg_loss(loss_mat=entropy, loss_mask=response_mask_tmp, loss_agg_mode=loss_agg_mode)
        entropy_low = agg_loss(loss_mat=entropy, loss_mask=mask_entropy_less_than_threshold, loss_agg_mode=loss_agg_mode)
        cov_high_entropy = agg_loss(loss_mat=cov, loss_mask=response_mask_tmp, loss_agg_mode=loss_agg_mode)
        cov_low_entropy = agg_loss(loss_mat=cov, loss_mask=mask_entropy_less_than_threshold, loss_agg_mode=loss_agg_mode)
        ratio_high_entropy = agg_loss(loss_mat=ratio, loss_mask=response_mask_tmp, loss_agg_mode=loss_agg_mode)
        ratio_low_entropy = agg_loss(loss_mat=ratio, loss_mask=mask_entropy_less_than_threshold, loss_agg_mode=loss_agg_mode)
    else:
        advantages_high_entropy = torch.tensor(0.0, device=get_device_id())
        advantages_low_entropy = torch.tensor(0.0, device=get_device_id())
        entropy_high = torch.tensor(0.0, device=get_device_id())
        entropy_low = torch.tensor(0.0, device=get_device_id())
        cov_high_entropy = torch.tensor(0.0, device=get_device_id())
        cov_low_entropy = torch.tensor(0.0, device=get_device_id())
        ratio_high_entropy = torch.tensor(0.0, device=get_device_id())
        ratio_low_entropy = torch.tensor(0.0, device=get_device_id())
        
    pg_loss, pg_clipfrac, ppo_kl = compute_policy_loss(old_log_prob=old_log_prob_tmp,
                                                            log_prob=log_prob,
                                                            advantages=advantages_tmp,
                                                            eos_mask=response_mask_tmp,
                                                            clip_ratio_high=0.28,
                                                            clip_ratio_low=0.2)
    
    ret_dict = compute_token_on_off_policy_loss(
        old_log_prob=old_log_prob_tmp, 
        log_prob=log_prob, 
        advantages=advantages_tmp, 
        eos_mask=response_mask_tmp, 
        cliprange=0.2, 
        clip_upper_bound=1.0,
        prefix_mask=off_policy_mask, 
        off_cliprange=0.3, 
        off_normalize=False, 
        off_abs_cliprange=-1, 
        off_max_clip=-1, 
        off_min_clip=-1,
        all_max_clip=-1, 
        off_policy_reshape="no_reshape", 
        off_policy_reshape_weight=1.0, 
        off_policy_reshape_pow_exp=0.5,
        on_policy_reshape="no_reshape", 
        on_policy_reshape_weight=1.0,
        on_policy_reshape_pow_exp=0.5,
        target_probs=None,
        loss_remove_token_mean=False,
        loss_remove_clip=False,
    )
    pg_loss2 = ret_dict['pg_loss']
    off_pg_loss2 = ret_dict['off_pg_loss']
    on_pg_loss2 = ret_dict['on_pg_loss']
    off_pg_clipfrac2 = ret_dict['off_pg_clipfrac']
    pg_clipfrac2 = ret_dict['on_pg_clipfrac']
    ppo_kl2 = ret_dict['ppo_kl']
    
    data = {
        'actor/off_pg_loss': off_pg_loss2.detach().item(),
        'actor/on_pg_loss': on_pg_loss2.detach().item(),
        'actor/off_pg_clipfrac': off_pg_clipfrac2.detach().item(),
    }
    if 'off_policy_prob' in ret_dict:
        data['actor/off_policy_prob'] = ret_dict['off_policy_prob'].detach().item()
    if 'on_policy_prob' in ret_dict:
        data['actor/on_policy_prob'] = ret_dict['on_policy_prob'].detach().item()
    if 'off_ratio_mean' in ret_dict:
        data['actor/off_ratio_mean'] = ret_dict['off_ratio_mean'].detach().item()
    if 'off_ratio_max_clip_frac' in ret_dict:
        data['actor/off_ratio_max_clip_frac'] = ret_dict['off_ratio_max_clip_frac'].detach().item()
    if 'off_ratio_min_clip_frac' in ret_dict:
        data['actor/off_ratio_min_clip_frac'] = ret_dict['off_ratio_min_clip_frac'].detach().item()
    
    print(ret_dict)
    print(off_policy_mask)
    import ipdb;ipdb.set_trace()

    response_mask_tmp_sum = response_mask_tmp.sum(axis=None)
    pg_loss = pg_loss* response_mask_tmp_sum
    pg_clipfrac = pg_clipfrac* response_mask_tmp_sum / response_mask_sum
    ppo_kl = ppo_kl* response_mask_tmp_sum / response_mask_sum

    # policy_loss = pg_loss / response_mask_sum

    # compute entropy loss from entropy
    entropy_coeff = 0.001
    if entropy_coeff != 0:
        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask_tmp, loss_agg_mode=loss_agg_mode)

        # compute policy loss
        policy_loss = pg_loss / response_mask_sum - entropy_loss * entropy_coeff
    else:
        policy_loss = pg_loss / response_mask_sum
        entropy_loss = torch.zeros_like(policy_loss)

    gradient_accumulation = 32 // 8
    loss = policy_loss / gradient_accumulation




    pg_loss_, pg_clipfrac_, ppo_kl_ =compute_policy_loss_weighted(old_log_prob=old_log_prob_tmp,
                                                                log_prob=log_prob,
                                                                advantages=advantages_tmp,
                                                                eos_mask=response_mask_tmp,
                                                                importance_weights=importance_weights_tmp,
                                                                clip_ratio_high=0.28,
                                                                clip_ratio_low=0.2)

    response_mask_tmp_sum_ = response_mask_tmp.sum(axis=None)
    pg_loss_ = pg_loss_* response_mask_tmp_sum_
    pg_clipfrac_ = pg_clipfrac_* response_mask_tmp_sum_ / response_mask_sum
    ppo_kl_ = ppo_kl_* response_mask_tmp_sum_ / response_mask_sum

    # policy_loss = pg_loss / response_mask_sum

    # compute entropy loss from entropy
    entropy_coeff = 0.001
    if entropy_coeff != 0:
        entropy_loss_ = agg_loss(loss_mat=entropy, loss_mask=response_mask_tmp, loss_agg_mode=loss_agg_mode)

        # compute policy loss
        policy_loss_ = pg_loss_ / response_mask_sum - entropy_loss_ * entropy_coeff
    else:
        policy_loss_ = pg_loss_ / response_mask_sum
        entropy_loss_ = torch.zeros_like(policy_loss)

    gradient_accumulation = 32 // 8
    loss_ = policy_loss_ / gradient_accumulation




    # 将重要性权重与response_mask结合
    combined_mask = response_mask_tmp * importance_weights_tmp

    # 计算weighted response_mask_sum用于归一化
    weighted_response_mask_sum = combined_mask.sum(axis=None)

    pg_loss__, pg_clipfrac__, ppo_kl__ =compute_policy_loss_weighted(old_log_prob=old_log_prob_tmp,
                                                                log_prob=log_prob,
                                                                advantages=advantages_tmp,
                                                                eos_mask=response_mask_tmp,
                                                                importance_weights=importance_weights_tmp,
                                                                clip_ratio_high=0.28,
                                                                clip_ratio_low=0.2)

    if importance_weights_tmp is not None:
        # 如果使用重要性权重，loss已经在函数内部正确计算
        response_mask_tmp_sum = (response_mask_tmp * importance_weights_tmp).sum(axis=None)
        response_mask_sum_total = (response_mask * importance_weights_tmp).sum(axis=None)
    else:
        # 如果没有重要性权重，使用原始逻辑
        response_mask_tmp_sum = response_mask_tmp.sum(axis=None)
        response_mask_sum_total = response_mask_sum

    response_mask_tmp_sum_ = response_mask_tmp.sum(axis=None)
    pg_loss__ = pg_loss__* response_mask_tmp_sum_
    pg_clipfrac__ = pg_clipfrac__* response_mask_tmp_sum_ / response_mask_sum
    ppo_kl__ = ppo_kl__* response_mask_tmp_sum_ / response_mask_sum

    # policy_loss = pg_loss / response_mask_sum

    # compute entropy loss from entropy
    entropy_coeff = 0.001
    if entropy_coeff != 0:
        entropy_loss__ = agg_loss(loss_mat=entropy, loss_mask=response_mask_tmp, loss_agg_mode=loss_agg_mode)

        # compute policy loss
        policy_loss__ = pg_loss__ / response_mask_sum - entropy_loss__ * entropy_coeff
    else:
        policy_loss__ = pg_loss__ / response_mask_sum
        entropy_loss_ = torch.zeros_like(policy_loss)

    gradient_accumulation = 32 // 8
    loss__ = policy_loss__ / gradient_accumulation

    import ipdb;ipdb.set_trace()
    
    print(xx, policy_loss, policy_loss_, policy_loss__)