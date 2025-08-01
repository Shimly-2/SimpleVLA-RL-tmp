import torch
import numpy as np

def _sample_indices_by_probability(indices, sampling_probs, num_samples):
    """
    根据概率采样索引
    
    Args:
        sampling_probs: 采样概率
        num_samples: 需要采样的数量
        
    Returns:
        sampled_indices: 采样的索引
    """
    buffer_size = 15
    num_samples = min(num_samples, buffer_size)
    
    # 使用numpy进行概率采样
    # indices = np.arange(buffer_size)
    sampled_indices = np.random.choice(
        indices, 
        size=num_samples, 
        replace=False, 
        p=sampling_probs.numpy()
    )
    
    return sampled_indices

def _compute_sampling_probabilities(sampling_strategy='adv_priority', batch_id2mean=[], batch_id2std=[], advantages=[], version_diffs=[]):
    """
    计算所有buffer中数据的采样概率
    
    Returns:
        sampling_probs: torch.Tensor, shape (buffer_size,)
    """
    # if not self.buffer:
    #     return torch.tensor([])
    
    if sampling_strategy == 'adv_priority':
        # 基于advantage计算优先级
        scores = torch.tensor(advantages, dtype=torch.float32)
        priority_weights = torch.pow(scores, 0.6)
        sampling_probs = priority_weights / torch.sum(priority_weights) # (bs * sample)
    
    elif sampling_strategy == 'mean_reward_priority':
        # 基于质量分数和新鲜度计算优先级
        scores = []
        for id2mean, version_diff in zip(batch_id2mean, version_diffs):
            freshness_score = 1.0 / (1.0 + version_diff)
            
            # 组合分数：70%质量 + 30%新鲜度
            combined_score = 0.3 * id2mean + 1.0 * freshness_score
            scores.append(combined_score)
        # import ipdb;ipdb.set_trace()
        print(scores)
        # 转换为采样概率
        scores = torch.tensor(scores, dtype=torch.float32)
        priority_weights = torch.pow(scores, 0.6)
        sampling_probs = priority_weights / torch.sum(priority_weights) # (bs)
        
    elif sampling_strategy == 'std_reward_priority':
        # 基于质量分数和新鲜度计算优先级
        scores = []
        for id2std, version_diff in zip(batch_id2std, version_diffs):
            freshness_score = 1.0 / (1.0 + 0.1 * version_diff)
            
            # 组合分数：70%质量 + 30%新鲜度
            combined_score = 0.5 * id2std + 0.5 * freshness_score
            scores.append(combined_score)
        
        # 转换为采样概率
        scores = torch.tensor(scores, dtype=torch.float32)
        priority_weights = torch.pow(scores, 0.6)
        sampling_probs = priority_weights / torch.sum(priority_weights) # (bs)
        
    elif sampling_strategy == 'recent':
        # 基于时间戳的概率（越新概率越高）
        timestamps = [item['timestamp'] for item in self.buffer]
        max_timestamp = max(timestamps)
        
        # 计算相对新鲜度
        freshness_scores = []
        for ts in timestamps:
            age = max_timestamp - ts
            freshness = np.exp(-age / 3600)  # 1小时衰减
            freshness_scores.append(max(freshness, 0.01))
        
        freshness_scores = torch.tensor(freshness_scores, dtype=torch.float32)
        sampling_probs = freshness_scores / torch.sum(freshness_scores)
        
    elif sampling_strategy == 'uniform':
        # 均匀概率
        buffer_size = len(self.buffer)
        sampling_probs = torch.ones(buffer_size, dtype=torch.float32) / buffer_size
        
    elif sampling_strategy == 'xx':  # mixed or other
        # 混合策略：基于质量和新鲜度
        quality_scores = [max(item['quality_score'], 0.01) for item in self.buffer]
        timestamps = [item['timestamp'] for item in self.buffer]
        max_timestamp = max(timestamps) if timestamps else 0
        
        combined_scores = []
        for q_score, ts in zip(quality_scores, timestamps):
            age = max_timestamp - ts
            freshness = np.exp(-age / 3600)
            combined = 0.6 * q_score + 0.4 * freshness
            combined_scores.append(combined)
        
        combined_scores = torch.tensor(combined_scores, dtype=torch.float32)
        priority_weights = torch.pow(combined_scores, self.priority_alpha)
        sampling_probs = priority_weights / torch.sum(priority_weights)
        
    elif sampling_strategy == 'version_only_with_mean_tiebreak':
        # 方案4：主要看version_diff，mean只用于同version_diff的排序
        scores = []
        
        # 按version_diff分组
        version_groups = {}
        for i, (mean_val, version_diff) in enumerate(zip(batch_id2mean, version_diffs)):
            if version_diff not in version_groups:
                version_groups[version_diff] = []
            version_groups[version_diff].append((i, mean_val))
        
        # 为每个version_diff分配基础分数
        base_scores = {}
        sorted_versions = sorted(version_groups.keys())
        for i, version in enumerate(sorted_versions):
            # version_diff越小，基础分数越高
            base_scores[version] = len(sorted_versions) - i
        
        # 计算最终分数
        final_scores = [0.0] * len(batch_id2mean)
        for version_diff, items in version_groups.items():
            base_score = base_scores[version_diff]
            
            # 在同一版本组内，根据mean_reward排序
            for idx, mean_val in items:
                final_scores[idx] = base_score + 0.3 * mean_val  # mean作为微调
        
        # print("方案4 - version主导 + mean微调:")
        # print(f"Scores: {final_scores}")
        
        scores = torch.tensor(final_scores, dtype=torch.float32)
        sampling_probs = scores / torch.sum(scores)  # 直接归一化，不使用power
        
        # priority_weights = torch.pow(scores, 0.9)
        # sampling_probs = priority_weights / torch.sum(priority_weights) # (bs)
    
    return sampling_probs


batch_id2mean = [1, 0.25, 0, 0.875, 0.5, 1, 0.25, 0, 0.875, 0.5, 1, 0.25, 0, 0.875, 0.5]
version_diffs = [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
print(batch_id2mean)
print(version_diffs)
sampling_probs = _compute_sampling_probabilities(sampling_strategy='version_only_with_mean_tiebreak', batch_id2mean=batch_id2mean, batch_id2std=[], advantages=[], version_diffs=version_diffs)

print(sampling_probs)

result = _sample_indices_by_probability(indices, sampling_probs, num_samples=5)


print(result)

import ipdb;ipdb.set_trace()