# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict
import verl
import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns

def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor,
                                                  gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++. 
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns

def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor,
                                    eos_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward 
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores

# def compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index, epsilon=1e-6, 
#                                         num_quantiles=32):
#     """
#     分布式GRPO优势计算
    
#     Args:
#         num_quantiles: 分位数数量
#     """
#     response_length = token_level_rewards.shape[-1]
#     scores = token_level_rewards.sum(dim=-1)
    
#     # 1. 构建分位数分布
#     id2score_distribution = defaultdict(list)
#     id2quantiles = {}
    
#     with torch.no_grad():
#         bsz = scores.shape[0]
#         for i in range(bsz):
#             id2score_distribution[index[i]].append(scores[i])
        
#         # 2. 计算每组的分位数
#         for idx in id2score_distribution:
#             scores_tensor = torch.tensor(id2score_distribution[idx])
#             if len(scores_tensor) >= num_quantiles:
#                 # 计算分位数
#                 quantiles = torch.quantile(scores_tensor, 
#                                          torch.linspace(0, 1, num_quantiles))
#                 id2quantiles[idx] = quantiles
#             else:
#                 # 数据不足时使用均值和标准差
#                 mean = scores_tensor.mean()
#                 std = scores_tensor.std()
#                 # 构造正态分布的分位数
#                 normal_quantiles = torch.linspace(-2, 2, num_quantiles)
#                 quantiles = mean + std * normal_quantiles
#                 id2quantiles[idx] = quantiles
        
#         # 3. 基于分布的优势计算
#         distributional_advantages = torch.zeros_like(token_level_rewards)
        
#         for i in range(bsz):
#             current_score = scores[i]
#             quantiles = id2quantiles[index[i]]
            
#             # 计算当前分数在分布中的位置
#             percentile = compute_percentile_rank(current_score, quantiles)
            
#             # 分布优势 = 2 * (百分位 - 0.5)，范围在[-1, 1]
#             dist_advantage = 2 * (percentile - 0.5)
            
#             # 扩展到整个序列
#             distributional_advantages[i] = dist_advantage * eos_mask[i]
    
#     return distributional_advantages, distributional_advantages

def compute_percentile_rank(score, quantiles):
    """计算分数在分位数中的百分位排名"""
    # 找到分数在分位数中的位置
    rank = torch.searchsorted(quantiles, score, right=True)
    percentile = rank.float() / len(quantiles)
    return torch.clamp(percentile, 0.0, 1.0)



def compute_rloo_returns(data:verl.DataProto, eos_mask:torch.Tensor,n_samples,config):
    # calculate rloo reward on different reward sources, and sum again
    with torch.no_grad():
        # reward = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        advantages = torch.zeros_like(data.batch['responses'],dtype=torch.float32)
        discount_rewards=[]
        for k,v in data.batch.items():
            if k == 'rm_scores':
                gamma = config.algorithm.adv_params.reward_model_gamma
                reward_mask = eos_mask.bool()
            elif k == 'gt_scores':
                gamma = config.algorithm.adv_params.verifier_gamma
                prompt_ids = data.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(-1)
                reward_mask = torch.zeros_like(v, dtype=torch.bool)
                reward_mask[torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device), valid_response_length-1]=True
            else: # not a reward tensor
                continue
            reward_tensor = v.clone()
            reward_tensor[~reward_mask]=0
            for start_pos in range(0, reward_tensor.shape[0], n_samples):
                cur_rewards_mean = torch.cat(
                    [ reward_tensor[pos:pos + 1][ reward_mask[pos:pos + 1] ].mean(dim=0, keepdim=True) for pos
                     in range(start_pos, start_pos + n_samples) ], dim=0)
                cur_rewards_sum = cur_rewards_mean.sum()
                cur_reward_baseline = cur_rewards_sum / (n_samples - 1)
                reward_tensor[start_pos:start_pos + n_samples][
                    reward_mask[start_pos:start_pos + n_samples]] = \
                    reward_tensor[start_pos:start_pos + n_samples][
                        reward_mask[start_pos:start_pos + n_samples]] * (
                                n_samples / (n_samples - 1)) - cur_reward_baseline

            discount_reward = torch.zeros_like(reward_tensor)
            for step in reversed(range(reward_tensor.shape[1])):
                if step == reward_tensor.shape[1]-1:
                    discount_reward[:,step] = reward_tensor[:, step]
                else:
                    discount_reward[:,step] = reward_tensor[:, step] + gamma * discount_reward[:, step+1]
            discount_rewards.append(discount_reward)
        # return is the sum of discounted reward
        returns = sum(discount_rewards)
        # advantage is whitened return
        advantages = returns.clone()
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns

def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


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
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
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
    ppo_kl = verl_F.masked_mean_weighted(-negative_approx_kl, eos_mask, importance_weights)

    # 计算policy loss
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)

    # 使用带权重的masked mean
    pg_loss = verl_F.masked_mean_weighted(torch.max(pg_losses, pg_losses2), eos_mask, importance_weights)
    pg_clipfrac = verl_F.masked_mean_weighted(torch.gt(pg_losses2, pg_losses).float(), eos_mask, importance_weights)
    
    return pg_loss, pg_clipfrac, ppo_kl

def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss



def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError

def compute_ce_dpo_loss_rm(token_level_scores, acc, eos_mask, beta):
    cur_scores=((token_level_scores*eos_mask).sum(dim=1)*beta).sigmoid()
    cur_dpo_loss=torch.nn.functional.binary_cross_entropy(cur_scores, acc)
    return cur_dpo_loss

def compute_dpo_accuracy(token_level_scores, acc, eos_mask, n_samples):
    dpo_acc=[]
    for start_id in range(0, token_level_scores.shape[0], n_samples):
        cur_scores=(token_level_scores[start_id:start_id+n_samples]*eos_mask[start_id:start_id+n_samples]).sum(dim=1)
        def get_upper_triangle(tensor_x):
            diff_matrix = tensor_x.unsqueeze(1)-tensor_x.unsqueeze(0)
            upper_tri_indices=torch.triu(torch.ones_like(diff_matrix).bool(), diagonal=1)
            return diff_matrix[upper_tri_indices]

        cur_acc_diff=get_upper_triangle(acc[start_id:start_id+n_samples] ) # in range [-1,1]
        cur_score_diff=get_upper_triangle(cur_scores) # in R
        cur_score_prediction= (cur_score_diff>0).float() # in [0,1]

        # print(f"{token_level_scores=}")
        # print(f"{acc=}")
        # print(f"{cur_scores=}")
        # print(f"{cur_score_diff=}")
        
        if cur_acc_diff.abs().sum()==0:
            cur_acc=torch.zeros_like(cur_score_prediction[0])+0.5
        else:
            cur_acc= (((cur_score_diff > 0) == (cur_acc_diff > 0)).float() * cur_acc_diff.abs()).sum()/cur_acc_diff.abs().sum()

        dpo_acc.append(cur_acc.unsqueeze(0))

    return torch.cat(dpo_acc, dim=0).mean()

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
        loss = verl_F.masked_mean(loss_mat, loss_mask)
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
        loss = verl_F.masked_mean_weighted(loss_mat, loss_mask, importance_weights)
        
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
            seq_loss = verl_F.masked_mean_weighted(seq_loss_mat, seq_mask, seq_weights)
            seq_losses.append(seq_loss)
        
        # 对所有序列取平均
        loss = torch.mean(torch.stack(seq_losses))
        
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss

def compute_sft_pure_loss(log_prob, eos_mask):
    sft_losses = -log_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return sft_loss

def compute_grpo_outcome_advantage_split(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   on_policy_mask: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   use_std: bool = True):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            # only include on-policy samples for mean and std calculation
            if on_policy_mask[i].item() is True:
                id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        # process std
        for idx in id2std:
            if id2std[idx].item() == 0:
                id2std[idx] = torch.tensor(1.0)
        for i in range(bsz):
            if use_std:
                try:
                    scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
                except:
                    pass
            else:
                scores[i] = (scores[i] - id2mean[index[i]])
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores

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
    
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

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
        on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)
        on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
    else:
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
        on_pg_clipfrac = torch.tensor(0.0)
    
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
        off_ratio = off_ratio * prefix_mask
        # assert ((target_probs > 0) == prefix_mask).all()
        
    # clip off-policy ratio
    if off_max_clip is not None:
        off_ratio = torch.clamp(off_ratio, max=off_max_clip)
        off_ratio_max_clip_frac = verl_F.masked_mean((off_ratio == off_max_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_max_clip_frac = torch.tensor(0.0)
        
    if off_min_clip is not None:
        off_ratio = torch.clamp(off_ratio, min=off_min_clip)
        off_ratio_min_clip_frac = verl_F.masked_mean((off_ratio == off_min_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_min_clip_frac = torch.tensor(0.0)

    off_ratio_mean = verl_F.masked_mean(off_ratio, prefix_mask * eos_mask)
    if off_ratio_mean.isnan().any().item():
        off_ratio_mean = torch.tensor(0.0)

    off_pg_losses = -advantages * off_ratio
    off_pg_loss = verl_F.masked_mean(off_pg_losses, prefix_mask * eos_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0)
    off_pg_clipfrac = torch.tensor(0.0)
    
    prefix_mask = prefix_mask.float()
    pg_losses = off_pg_losses * prefix_mask + on_pg_losses * (1 - prefix_mask)
    
    # log on/off probs
    off_policy_probs = torch.exp(log_prob)
    off_policy_prob = verl_F.masked_mean(off_policy_probs, prefix_mask * eos_mask)
    if off_policy_prob.isnan().item() is True:
        off_policy_prob = torch.tensor(0.0)
    on_policy_probs = torch.exp(old_log_prob)
    on_policy_prob = verl_F.masked_mean(on_policy_probs, (1.0-prefix_mask) * eos_mask)
    if on_policy_prob.isnan().item() is True:
        on_policy_prob = torch.tensor(0.0)
            
    if all_max_clip is not None:
        p_on = torch.exp(log_prob)
        p_on_mask = (p_on <= all_max_clip).float()
        eos_mask = eos_mask * p_on_mask
        pg_losses = pg_losses * p_on_mask
        
    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
        print(f'no token mean: mean normalization {eos_mask.shape[-1]}')
    else:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)

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
    
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

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
        on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)
        on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
    else:
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
        on_pg_clipfrac = torch.tensor(0.0)
    
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
        off_ratio = off_ratio * prefix_mask
        # assert ((target_probs > 0) == prefix_mask).all()
        
    # clip off-policy ratio
    if off_max_clip is not None:
        off_ratio = torch.clamp(off_ratio, max=off_max_clip)
        off_ratio_max_clip_frac = verl_F.masked_mean((off_ratio == off_max_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_max_clip_frac = torch.tensor(0.0)
        
    if off_min_clip is not None:
        off_ratio = torch.clamp(off_ratio, min=off_min_clip)
        off_ratio_min_clip_frac = verl_F.masked_mean((off_ratio == off_min_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_min_clip_frac = torch.tensor(0.0)

    off_ratio_mean = verl_F.masked_mean(off_ratio, prefix_mask * eos_mask)
    if off_ratio_mean.isnan().any().item():
        off_ratio_mean = torch.tensor(0.0)

    off_pg_losses = -advantages * off_ratio
    off_pg_loss = verl_F.masked_mean(off_pg_losses, prefix_mask * eos_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0)
    off_pg_clipfrac = torch.tensor(0.0)
    
    prefix_mask = prefix_mask.float()
    pg_losses = off_pg_losses * prefix_mask + on_pg_losses * (1 - prefix_mask)
    
    # log on/off probs
    off_policy_probs = torch.exp(log_prob)
    off_policy_prob = verl_F.masked_mean(off_policy_probs, prefix_mask * eos_mask)
    if off_policy_prob.isnan().item() is True:
        off_policy_prob = torch.tensor(0.0)
    on_policy_probs = torch.exp(old_log_prob)
    on_policy_prob = verl_F.masked_mean(on_policy_probs, (1.0-prefix_mask) * eos_mask)
    if on_policy_prob.isnan().item() is True:
        on_policy_prob = torch.tensor(0.0)
            
    if all_max_clip is not None:
        p_on = torch.exp(log_prob)
        p_on_mask = (p_on <= all_max_clip).float()
        eos_mask = eos_mask * p_on_mask
        pg_losses = pg_losses * p_on_mask
        
    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
        print(f'no token mean: mean normalization {eos_mask.shape[-1]}')
    else:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)

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
    
def compute_token_on_off_policy_loss_v2(
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
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

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
        upper_bound = max(1.0 + clip_upper_bound, 1.0 + cliprange)
        if upper_bound == clip_upper_bound:
            print('clip upper bound is used: ', clip_upper_bound)
            
        on_ratio_mean = verl_F.masked_mean(ratio, eos_mask)
        if on_ratio_mean.isnan().any().item():
            on_ratio_mean = torch.tensor(0.0)

        if loss_remove_clip is False:
            on_pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, upper_bound)
            on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)
            on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
            on_pg_loss = verl_F.masked_mean(on_pg_losses, eos_mask)
            on_ratio_max_clip_frac = verl_F.masked_mean((torch.clamp(ratio, 1.0 - cliprange, upper_bound) == upper_bound).float(), eos_mask)
            on_ratio_min_clip_frac = verl_F.masked_mean((torch.clamp(ratio, 1.0 - cliprange, upper_bound) == 1.0 - cliprange).float(), eos_mask)
        else:
            on_pg_loss = verl_F.masked_mean(on_pg_losses, eos_mask)
            on_pg_clipfrac = torch.tensor(0.0)
            on_ratio_max_clip_frac = torch.tensor(0.0)
            on_ratio_min_clip_frac = torch.tensor(0.0)
            

        pg_losses = on_pg_loss.clone()
        off_pg_loss = torch.tensor(0.0)
        off_pg_clipfrac = torch.tensor(0.0)
        off_ratio_mean = torch.tensor(0.0)
        off_ratio_max_clip_frac = torch.tensor(0.0)
        off_ratio_min_clip_frac = torch.tensor(0.0)
        
        on_policy_probs = torch.exp(old_log_prob)
        on_policy_prob = verl_F.masked_mean(on_policy_probs, eos_mask)
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
            off_ratio = torch.clamp(off_ratio, max=1 + off_max_clip)
            off_ratio_max_clip_frac = verl_F.masked_mean((off_ratio == 1 + off_max_clip).float(), eos_mask)
        else:
            off_ratio_max_clip_frac = torch.tensor(0.0)
            
        if off_min_clip is not None:
            off_ratio = torch.clamp(off_ratio, min=1 - off_min_clip)
            off_ratio_min_clip_frac = verl_F.masked_mean((off_ratio == 1 - off_min_clip).float(), eos_mask)
        else:
            off_ratio_min_clip_frac = torch.tensor(0.0)

        off_ratio_mean = verl_F.masked_mean(off_ratio, eos_mask)
        if off_ratio_mean.isnan().any().item():
            off_ratio_mean = torch.tensor(0.0)

        off_pg_losses = -advantages * off_ratio
        off_pg_loss = verl_F.masked_mean(off_pg_losses, eos_mask)
        # off_pg_losses2 = -advantages * torch.clamp(off_ratio, 1.0 - 0.1, 1.0 + 0.1)
        # off_pg_clipfrac = verl_F.masked_mean(torch.gt(off_pg_losses2, off_pg_losses).float(), eos_mask)
        # off_pg_losses = torch.max(off_pg_losses, off_pg_losses2)
        # off_pg_loss = verl_F.masked_mean(off_pg_losses, eos_mask)
        if off_pg_loss.isnan().item() is True:
            off_pg_loss = torch.tensor(0.0)
        off_pg_clipfrac = torch.tensor(0.0)
        
        pg_losses = off_pg_loss.clone()
        on_pg_loss = torch.tensor(0.0)
        on_pg_clipfrac = torch.tensor(0.0)
        on_ratio_mean = torch.tensor(0.0)
        on_ratio_max_clip_frac = torch.tensor(0.0)
        on_ratio_min_clip_frac = torch.tensor(0.0)
        ppo_kl = torch.tensor(0.0)
        
        off_policy_probs = torch.exp(log_prob)
        off_policy_prob = verl_F.masked_mean(off_policy_probs, eos_mask)
        if off_policy_prob.isnan().item() is True:
            off_policy_prob = torch.tensor(0.0)
        on_policy_prob = torch.tensor(0.0)
    
    # prefix_mask = prefix_mask.float()
    # pg_losses = off_pg_losses * prefix_mask + on_pg_losses * (1 - prefix_mask)
    
    # log on/off probs
      
    # if all_max_clip is not None:
    #     p_on = torch.exp(log_prob)
    #     p_on_mask = (p_on <= all_max_clip).float()
    #     eos_mask = eos_mask * p_on_mask
    #     pg_losses = pg_losses * p_on_mask
        
    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
        print(f'no token mean: mean normalization {eos_mask.shape[-1]}')
    else:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)

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
        "on_ratio_mean": on_ratio_mean,
        "on_ratio_max_clip_frac": on_ratio_max_clip_frac,
        "on_ratio_min_clip_frac": on_ratio_min_clip_frac,
    }