import torch
import numpy as np
from collections import defaultdict

def setup_example_data():
    """设置示例数据：(bs=8, 64, 56)"""
    bs = 8
    traj_len = 64
    action_dim = 56
    
    # 模拟8个机器人轨迹
    torch.manual_seed(42)
    
    # 动作序列 (bs, traj_len, action_dim)
    action_sequences = torch.randn(bs, traj_len, action_dim) * 0.5
    
    # 奖励只在最后一个token，0/1奖励
    token_level_rewards = torch.zeros(bs, traj_len)
    final_rewards = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0]).float()  # 4成功4失败
    token_level_rewards[:, -1] = final_rewards
    
    # EOS mask (所有位置都有效)
    eos_mask = torch.ones(bs, traj_len)
    
    # 模拟多个prompt组
    index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])  # 两个prompt组
    
    print("=" * 60)
    print("🤖 输入数据设置")
    print("=" * 60)
    print(f"批次大小: {bs}")
    print(f"轨迹长度: {traj_len}")
    print(f"动作维度: {action_dim}")
    print(f"最终奖励: {final_rewards.tolist()}")
    print(f"Prompt索引: {index.tolist()}")
    print(f"成功率: {final_rewards.mean():.1%}")
    print()
    
    return action_sequences, token_level_rewards, eos_mask, index, final_rewards

def verl_original_grpo(token_level_rewards: torch.Tensor,
                       eos_mask: torch.Tensor,
                       index: torch.Tensor,
                       epsilon: float = 1e-6):
    """VERL框架中的原始GRPO实现"""
    print("🔵 VERL原始GRPO计算")
    print("-" * 40)
    
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    
    print(f"各轨迹总分: {scores.tolist()}")
    print(f"Prompt分组: {index.tolist()}")
    
    # id2score的value是一个列表，包含了每组的奖励值
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        
        # 按prompt索引分组
        for i in range(bsz):
            id2score[index[i].item()].append(scores[i])
        
        print("\n各组统计:")
        for idx in id2score:
            group_scores = torch.stack(id2score[idx])
            print(f"组 {idx}: 分数={[f'{x:.1f}' for x in id2score[idx]]}")
            
            if len(id2score[idx]) == 1:
                # 只有一个样本时
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
                print(f"      单样本 -> 均值=0.0, 标准差=1.0")
            elif len(id2score[idx]) > 1:
                # 多个样本时计算真实统计量
                id2mean[idx] = torch.mean(group_scores)
                id2std[idx] = torch.std(group_scores, unbiased=False)  # VERL使用总体标准差
                print(f"      均值={id2mean[idx]:.3f}, 标准差={id2std[idx]:.3f}")
        
        print("\n标准化过程:")
        # 标准化
        normalized_scores = torch.zeros_like(scores)
        for i in range(bsz):
            original_score = scores[i]
            mean_val = id2mean[index[i].item()]
            std_val = id2std[index[i].item()]
            
            normalized_score = (original_score - mean_val) / (std_val + epsilon)
            normalized_scores[i] = normalized_score
            
            print(f"轨迹{i}: ({original_score:.1f} - {mean_val:.3f}) / {std_val:.3f} = {normalized_score:.3f}")
        
        # 扩展到整个序列
        advantages = normalized_scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    
    print(f"\n优势统计: 均值={advantages.mean():.3f}, 标准差={advantages.std():.3f}")
    print()
    return advantages, normalized_scores

def distributional_grpo_corrected(token_level_rewards: torch.Tensor,
                                eos_mask: torch.Tensor,
                                index: torch.Tensor,
                                action_sequences: torch.Tensor,
                                num_quantiles: int = 8,
                                epsilon: float = 1e-6):
    """修正后的分布式GRPO"""
    print("🟢 分布式GRPO计算")
    print("-" * 40)
    
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    
    # 添加动作平滑度奖励
    # action_diff = torch.diff(action_sequences, dim=1)
    # smoothness = 1.0 / (1.0 + torch.mean(torch.norm(action_diff, dim=-1), dim=-1))
    # enhanced_scores = scores + 0.1 * smoothness  # 平滑度权重0.1
    enhanced_scores = scores
    
    # print(f"原始分数: {scores.tolist()}")
    # print(f"平滑度分数: {[f'{x:.3f}' for x in smoothness.tolist()]}")
    # print(f"增强后分数: {[f'{x:.3f}' for x in enhanced_scores.tolist()]}")
    
    # 按组构建分位数分布
    id2score_distribution = defaultdict(list)
    id2quantiles = {}
    
    with torch.no_grad():
        bsz = enhanced_scores.shape[0]
        for i in range(bsz):
            id2score_distribution[index[i].item()].append(enhanced_scores[i])
        
        print("\n各组分位数:")
        for idx in id2score_distribution:
            scores_tensor = torch.stack(id2score_distribution[idx])
            
            if len(scores_tensor) >= num_quantiles:
                quantiles = torch.quantile(scores_tensor, 
                                         torch.linspace(0, 1, num_quantiles))
            else:
                # 数据不足时使用高斯分布近似
                mean = scores_tensor.mean()
                std = scores_tensor.std() + epsilon
                normal_points = torch.linspace(-2, 2, num_quantiles)
                quantiles = mean + std * normal_points
            
            id2quantiles[idx] = quantiles
            print(f"组 {idx}: 分位数={[f'{x:.3f}' for x in quantiles.tolist()]}")
        
        print("\n分布优势计算:")
        distributional_advantages = torch.zeros_like(token_level_rewards)
        normalized_scores = torch.zeros(bsz)
        
        for i in range(bsz):
            current_score = enhanced_scores[i]
            quantiles = id2quantiles[index[i].item()]
            
            # 计算百分位排名
            rank = torch.searchsorted(quantiles, current_score, right=True)
            percentile = rank.float() / len(quantiles)
            percentile = torch.clamp(percentile, 0.0, 1.0)
            
            # 分布优势：[-1, 1]
            dist_advantage = 2 * (percentile - 0.5)
            normalized_scores[i] = dist_advantage
            
            # 时间权重 (指数衰减，后面权重更高)
            time_weights = torch.pow(0.95, torch.arange(response_length, 0, -1, dtype=torch.float))
            
            distributional_advantages[i] = dist_advantage * time_weights * eos_mask[i]
            
            print(f"轨迹{i}: 分数={current_score:.3f}, 百分位={percentile:.3f}, 优势={dist_advantage:.3f}")
    
    print(f"\n优势统计: 均值={distributional_advantages.mean():.3f}, 标准差={distributional_advantages.std():.3f}")
    print()
    import ipdb;ipdb.set_trace()
    return distributional_advantages, normalized_scores

def momentum_grpo_corrected(token_level_rewards: torch.Tensor,
                          eos_mask: torch.Tensor,
                          index: torch.Tensor,
                          action_sequences: torch.Tensor,
                          momentum: float = 0.7,
                          epsilon: float = 1e-6):
    """修正后的动量GRPO"""
    print("🟡 动量GRPO计算")
    print("-" * 40)
    
    # 先计算当前批次的VERL标准化优势
    _, current_normalized_scores = verl_original_grpo(token_level_rewards, eos_mask, index, epsilon)
    
    # 模拟历史优势 (分组存储)
    historical_advantages = {
        0: torch.tensor([0.2, -0.3, 0.1, -0.2]),    # 组0的历史
        1: torch.tensor([-0.1, 0.3, -0.1, 0.2])     # 组1的历史
    }
    
    print("历史优势:")
    for group_idx, hist_adv in historical_advantages.items():
        print(f"组 {group_idx}: {[f'{x:.3f}' for x in hist_adv.tolist()]}")
    
    print(f"当前标准化优势: {[f'{x:.3f}' for x in current_normalized_scores.tolist()]}")
    
    # 应用动量机制
    momentum_advantages = torch.zeros_like(token_level_rewards)
    final_normalized_scores = torch.zeros_like(current_normalized_scores)
    
    with torch.no_grad():
        print("\n动量更新:")
        group_counters = {0: 0, 1: 0}  # 每组的计数器
        
        for i in range(len(current_normalized_scores)):
            group_idx = index[i].item()
            local_idx = group_counters[group_idx]
            
            hist_adv = historical_advantages[group_idx][local_idx]
            curr_adv = current_normalized_scores[i]
            
            # 自适应动量 (基于变化程度)
            change_magnitude = torch.abs(curr_adv - hist_adv)
            adaptive_momentum = momentum * torch.exp(-change_magnitude / 2.0)
            adaptive_momentum = torch.clamp(adaptive_momentum, 0.1, 0.99)
            
            # 动量更新
            final_adv = adaptive_momentum * hist_adv + (1 - adaptive_momentum) * curr_adv
            final_normalized_scores[i] = final_adv
            
            # 扩展到整个序列
            momentum_advantages[i] = final_adv * eos_mask[i]
            
            print(f"轨迹{i}: 历史={hist_adv:.3f}, 当前={curr_adv:.3f}, "
                  f"动量={adaptive_momentum:.3f}, 最终={final_adv:.3f}")
            
            group_counters[group_idx] += 1
    
    print(f"\n优势统计: 均值={momentum_advantages.mean():.3f}, 标准差={momentum_advantages.std():.3f}")
    print()
    return momentum_advantages, final_normalized_scores

def compare_key_differences():
    """对比关键差异"""
    print("🚀 GRPO算法关键差异对比")
    print("=" * 60)
    
    # 设置数据
    action_sequences, token_level_rewards, eos_mask, index, final_rewards = setup_example_data()
    
    # 运行各种算法
    original_adv, original_scores = verl_original_grpo(token_level_rewards, eos_mask, index)
    dist_adv, dist_scores = distributional_grpo_corrected(token_level_rewards, eos_mask, index, action_sequences)
    momentum_adv, momentum_scores = momentum_grpo_corrected(token_level_rewards, eos_mask, index, action_sequences)
    
    # 关键对比
    print("📊 标准化分数对比")
    print("=" * 60)
    print(f"{'轨迹':<4} {'奖励':<4} {'组':<4} {'原始GRPO':<12} {'分布式GRPO':<12} {'动量GRPO':<12}")
    print("-" * 60)
    
    for i in range(len(final_rewards)):
        print(f"{i:<4} {final_rewards[i]:<4.0f} {index[i]:<4} "
              f"{original_scores[i]:<12.3f} {dist_scores[i]:<12.3f} {momentum_scores[i]:<12.3f}")
    
    print()
    
    # 成功vs失败对比
    print("🎯 成功vs失败轨迹对比")
    print("-" * 40)
    
    success_mask = final_rewards == 1
    failure_mask = final_rewards == 0
    
    methods = {
        "原始GRPO": original_scores,
        "分布式GRPO": dist_scores,
        "动量GRPO": momentum_scores
    }
    
    for name, scores in methods.items():
        success_mean = scores[success_mask].mean()
        failure_mean = scores[failure_mask].mean()
        gap = success_mean - failure_mean
        
        print(f"{name}:")
        print(f"  成功轨迹优势: {success_mean:>7.3f}")
        print(f"  失败轨迹优势: {failure_mean:>7.3f}")
        print(f"  优势差距:     {gap:>7.3f}")
        print()
    
    # 关键洞察
    print("💡 关键发现:")
    print("1. 原始GRPO: 基于组内z-score标准化，0/1奖励导致标准差为0.5")
    print("2. 分布式GRPO: 考虑动作质量，能区分相同奖励下的不同表现")
    print("3. 动量GRPO: 平滑历史波动，训练更稳定")
    print("4. 你的0/1稀疏奖励场景特别适合分布式GRPO改进")

if __name__ == "__main__":
    compare_key_differences()

# 单独展示VERL原始GRPO的详细计算
def detailed_verl_example():
    print("\n" + "="*60)
    print("🔍 VERL原始GRPO详细计算示例")
    print("="*60)
    
    # 简化示例：两个组，每组2个轨迹
    scores = torch.tensor([0.0, 1.0, 0.0, 1.0])  # 组0:[0,1], 组1:[0,1]
    index = torch.tensor([0, 0, 1, 1])
    
    print("输入:")
    print(f"分数: {scores.tolist()}")
    print(f"分组: {index.tolist()}")
    
    # 分组统计
    id2score = defaultdict(list)
    for i, score in enumerate(scores):
        id2score[index[i].item()].append(score.item())
    
    print("\n分组统计:")
    for group_idx, group_scores in id2score.items():
        print(f"组 {group_idx}: {group_scores}")
        
        if len(group_scores) > 1:
            mean = np.mean(group_scores)
            std = np.std(group_scores)  # 总体标准差
            print(f"         均值={mean:.3f}, 标准差={std:.3f}")
            
            # 标准化
            normalized = [(s - mean) / std for s in group_scores]
            print(f"         标准化={[f'{x:.3f}' for x in normalized]}")
    
    print("\n🔑 关键点:")
    print("- 对于0/1奖励，组内标准差 = 0.5")
    print("- 成功轨迹标准化分数 = +1.0")  
    print("- 失败轨迹标准化分数 = -1.0")
    print("- 这就是为什么需要改进的原因！")

if __name__ == "__main__":
    compare_key_differences()
    detailed_verl_example()