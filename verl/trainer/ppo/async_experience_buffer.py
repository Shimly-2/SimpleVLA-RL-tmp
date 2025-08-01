import asyncio
import threading
import time
from collections import defaultdict, deque
from typing import Optional, List, Tuple, Dict, Optional, Callable, Any
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from dataclasses import dataclass
import numpy as np
import copy
from verl import DataProto

class AsyncExperienceBuffer:
    def __init__(self, config):
        self.config = config
        self.n_samples = config.data.n_samples
        self.train_batch_size = config.data.train_batch_size
        self.config.data.min_buffer_size = self.n_samples * self.train_batch_size
        
        # 缓冲区存储的是单个rollout，不是批次
        self.buffer = deque(maxlen=config.data.max_buffer_size)
        self.buffer_lock = threading.RLock()
        self.not_empty = threading.Condition(self.buffer_lock)
        self.not_full = threading.Condition(self.buffer_lock)
    
    def add_batch(self, batch_data: DataProto):
        """添加批次数据到缓冲区 - 将批次拆分为单个rollout"""
        with self.not_full:
            # 将批次拆分为单个rollout存储
            for i in range(len(batch_data.batch)):
                while len(self.buffer) >= self.config.data.max_buffer_size:
                    self.not_full.wait()  # 等待空间
                
                # 提取单个rollout
                single_rollout = batch_data[i:i+1]
                self.buffer.append(single_rollout)
            
            print(f"Added {len(batch_data.batch)} rollouts to buffer, "
                  f"buffer size: {len(self.buffer)}")
            
            self.not_empty.notify_all()  # 通知有新数据
    
    def sample_batch(self, batch_size: int = None) -> Optional[DataProto]:
        """从缓冲区采样批次 - 采样batch_size*n_samples个rollout组成训练批次"""
        # 如果没有指定batch_size，使用配置中的默认值
        if batch_size is None:
            batch_size = self.train_batch_size
        
        # 计算需要采样的rollout总数
        total_rollouts_needed = batch_size * self.n_samples
        
        with self.not_empty:
            # 等待足够的rollout用于训练
            while len(self.buffer) < total_rollouts_needed:
                if not self.not_empty.wait(timeout=1.0):
                    return None  # 超时返回None
            
            # 采样 batch_size * n_samples 个rollout
            sampled_rollouts = self._uniform_sample(total_rollouts_needed)
            
            if sampled_rollouts:
                # 合并成训练批次
                training_batch = DataProto.concat(sampled_rollouts)
                print(f"Sampled {len(training_batch.batch)} rollouts for training "
                      f"(batch_size={batch_size}, n_samples={self.n_samples}, "
                      f"total={total_rollouts_needed})")
                
                self.not_full.notify_all()  # 通知有空间
                return training_batch
            
            return None
        
    def sample_batch_grouped(self, batch_size: int = None) -> Optional[List[DataProto]]:
        """采样分组的批次 - 返回按原始样本分组的rollout列表"""
        if batch_size is None:
            batch_size = self.train_batch_size
        
        total_rollouts_needed = batch_size * self.n_samples
        
        with self.not_empty:
            while len(self.buffer) < total_rollouts_needed:
                if not self.not_empty.wait(timeout=1.0):
                    return None
            
            sampled_rollouts = self._uniform_sample(total_rollouts_needed)
            
            if sampled_rollouts:
                # 将rollout按组分组（每组n_samples个）
                grouped_rollouts = []
                for i in range(0, len(sampled_rollouts), self.n_samples):
                    group = sampled_rollouts[i:i + self.n_samples]
                    if len(group) == self.n_samples:  # 确保每组都有完整的n_samples
                        grouped_rollouts.append(group)
                
                print(f"Sampled {len(grouped_rollouts)} groups of {self.n_samples} rollouts each")
                self.not_full.notify_all()
                return grouped_rollouts
            
            return None
        
    def _uniform_sample(self, batch_size: int) -> List[DataProto]:
        """均匀采样rollout"""
        import random
        available_size = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), available_size)
        return [self.buffer[i] for i in indices]
    
    def get_rollout_groups_for_training(self, num_groups: int) -> List[DataProto]:
        """获取用于训练的rollout组 - 每组包含n_samples个来自同一原始样本的rollout"""
        with self.not_empty:
            # 这个方法可以用于需要保持rollout分组的场景
            # 例如，当我们需要比较同一prompt的不同response时
            pass
        
    def size(self) -> int:
        with self.buffer_lock:
            return len(self.buffer)
    
    def is_ready(self) -> bool:
        """检查缓冲区是否准备好进行训练"""
        min_rollouts_for_training = self.config.data.min_buffer_size
        # 确保至少有一个完整的训练批次
        min_for_batch = self.train_batch_size * self.n_samples
        return len(self.buffer) >= max(min_rollouts_for_training, min_for_batch)

class HERStrategy(Enum):
    """HER策略类型"""
    FINAL = "final"           # 使用episode的最终状态作为目标
    FUTURE = "future"         # 使用episode中未来的状态作为目标
    EPISODE = "episode"       # 使用episode中随机状态作为目标
    RANDOM = "random"         # 使用随机目标

@dataclass
class HERConfig:
    """HER配置"""
    enabled: bool = True
    strategy: HERStrategy = HERStrategy.FUTURE
    replay_k: int = 4  # 每个原始经验生成k个HER经验
    future_p: float = 1.0  # future策略中选择未来状态的概率
    reward_threshold: float = 0.1  # 低于此奖励的经验被认为是"失败"的
    max_episode_length: int = 512  # 最大episode长度
    goal_key: str = "goal"  # 目标在batch中的键名
    achieved_goal_key: str = "achieved_goal"  # 达成目标在batch中的键名

class AsyncExperienceBufferWithHER:
    def __init__(self, config, her_config: Optional[HERConfig] = None):
        self.config = config
        self.her_config = her_config or HERConfig()
        
        self.n_samples = config.data.n_samples
        self.train_batch_size = config.data.train_batch_size
        
        # 主缓冲区：存储原始经验
        self.original_buffer = deque(maxlen=config.max_buffer_size // 2)
        # HER缓冲区：存储重新标记的经验
        self.her_buffer = deque(maxlen=config.max_buffer_size // 2)
        
        # Episode缓冲区：用于构建完整episode
        self.episode_buffer = {}  # task_id -> episode data
        self.completed_episodes = deque(maxlen=1000)  # 存储完成的episode
        
        # 线程安全
        self.buffer_lock = threading.RLock()
        self.not_empty = threading.Condition(self.buffer_lock)
        self.not_full = threading.Condition(self.buffer_lock)
        
        # HER统计
        self.her_stats = {
            'total_her_generated': 0,
            'successful_episodes': 0,
            'failed_episodes': 0,
            'average_episode_length': 0,
            'her_sample_ratio': 0.0
        }
        
        print(f"HER Buffer initialized:")
        print(f"  Strategy: {self.her_config.strategy.value}")
        print(f"  Replay K: {self.her_config.replay_k}")
        print(f"  Original buffer size: {config.max_buffer_size // 2}")
        print(f"  HER buffer size: {config.max_buffer_size // 2}")
    
    def add_batch(self, batch_data: DataProto):
        """添加批次数据，同时生成HER经验"""
        with self.not_full:
            # 等待空间
            while self._total_size() >= self.config.max_buffer_size:
                self.not_full.wait()
            
            # 处理每个rollout
            for i in range(len(batch_data.batch)):
                single_rollout = batch_data[i:i+1]
                
                # 添加原始经验
                self.original_buffer.append(single_rollout)
                
                # 尝试构建episode并生成HER经验
                if self.her_config.enabled:
                    self._process_rollout_for_her(single_rollout)
            
            print(f"Added {len(batch_data.batch)} rollouts to buffer. "
                  f"Total: {self._total_size()} "
                  f"(Original: {len(self.original_buffer)}, HER: {len(self.her_buffer)})")
            
            self.not_empty.notify_all()
            
    def sample_batch(self, batch_size: int = None) -> Optional[DataProto]:
        """从缓冲区采样批次 - 支持HER经验混合采样，采样batch_size*n_samples个rollout"""
        if batch_size is None:
            batch_size = self.train_batch_size
        
        # 计算需要采样的rollout总数
        total_rollouts_needed = batch_size * self.n_samples
        
        with self.not_empty:
            total_size = self._total_size()
            
            while total_size < total_rollouts_needed:
                if not self.not_empty.wait(timeout=1.0):
                    return None
                total_size = self._total_size()
            
            # 混合采样：原始经验 + HER经验
            sampled_rollouts = self._mixed_sample(total_rollouts_needed)
            
            if sampled_rollouts:
                training_batch = DataProto.concat(sampled_rollouts)
                
                # 更新HER采样比例统计
                her_count = sum(1 for rollout in sampled_rollouts 
                               if rollout.meta_info and rollout.meta_info.get('is_her', False))
                self.her_stats['her_sample_ratio'] = her_count / len(sampled_rollouts)
                
                print(f"Sampled {len(training_batch.batch)} rollouts for training "
                      f"(batch_size={batch_size}, n_samples={self.n_samples}, "
                      f"total={total_rollouts_needed}, HER: {her_count}, Original: {len(sampled_rollouts) - her_count})")
                
                self.not_full.notify_all()
                return training_batch
            
            return None
    
    def _mixed_sample(self, total_rollouts_needed: int) -> List[DataProto]:
        """混合采样原始经验和HER经验"""
        sampled_rollouts = []
        
        # 计算采样比例
        her_ratio = 0.5 if len(self.her_buffer) > 0 else 0.0
        her_count = int(total_rollouts_needed * her_ratio)
        original_count = total_rollouts_needed - her_count
        
        # 从原始缓冲区采样
        if original_count > 0 and len(self.original_buffer) > 0:
            original_samples = self._sample_from_buffer(self.original_buffer, original_count)
            sampled_rollouts.extend(original_samples)
        
        # 从HER缓冲区采样
        if her_count > 0 and len(self.her_buffer) > 0:
            her_samples = self._sample_from_buffer(self.her_buffer, her_count)
            sampled_rollouts.extend(her_samples)
        
        # 如果总数不够，从较大的缓冲区补充
        while len(sampled_rollouts) < total_rollouts_needed:
            if len(self.original_buffer) >= len(self.her_buffer):
                additional = self._sample_from_buffer(self.original_buffer, 1)
            else:
                additional = self._sample_from_buffer(self.her_buffer, 1)
            
            if additional:
                sampled_rollouts.extend(additional)
            else:
                break
        
        return sampled_rollouts[:total_rollouts_needed]
    
    def _sample_from_buffer(self, buffer: deque, count: int) -> List[DataProto]:
        """从指定缓冲区采样"""
        if len(buffer) == 0:
            return []
        
        available_size = min(count, len(buffer))
        indices = random.sample(range(len(buffer)), available_size)
        return [buffer[i] for i in indices]
    
    def _total_size(self) -> int:
        """总缓冲区大小"""
        return len(self.original_buffer) + len(self.her_buffer)
    
    def size(self) -> int:
        """兼容性方法"""
        return self._total_size()
    
    def is_ready(self) -> bool:
        """检查缓冲区是否准备好进行训练"""
        min_rollouts_for_training = self.config.data.min_buffer_size
        # 确保至少有一个完整的训练批次
        min_for_batch = self.train_batch_size * self.n_samples
        return len(self.buffer) >= max(min_rollouts_for_training, min_for_batch)
    
    def _process_rollout_for_her(self, rollout: DataProto):
        """处理单个rollout以生成HER经验"""
        try:
            # 提取episode信息
            task_id = self._extract_task_id(rollout)
            if task_id is None:
                return
            
            # 检查rollout是否完成了episode
            is_episode_complete = self._is_episode_complete(rollout)
            
            # 将rollout添加到episode buffer
            if task_id not in self.episode_buffer:
                self.episode_buffer[task_id] = []
            
            self.episode_buffer[task_id].append(rollout)
            
            # 如果episode完成，生成HER经验
            if is_episode_complete:
                self._generate_her_for_episode(task_id)
                
        except Exception as e:
            print(f"HER processing error: {e}")
    
    def _extract_task_id(self, rollout: DataProto) -> Optional[str]:
        """提取rollout的task_id"""
        try:
            if 'task_id' in rollout.batch:
                return str(rollout.batch['task_id'][0].item())
            elif 'task_id' in rollout.non_tensor_batch:
                return str(rollout.non_tensor_batch['task_id'][0])
            else:
                # 使用uid作为fallback
                return str(rollout.non_tensor_batch.get('uid', [None])[0])
        except:
            return None
    
    def _is_episode_complete(self, rollout: DataProto) -> bool:
        """判断episode是否完成"""
        try:
            # 检查是否有结束标志
            if 'done' in rollout.batch:
                return bool(rollout.batch['done'][0].item())
            
            # 检查是否达到最大长度
            task_id = self._extract_task_id(rollout)
            if task_id and task_id in self.episode_buffer:
                return len(self.episode_buffer[task_id]) >= self.her_config.max_episode_length
            
            # 默认认为每个rollout都是完整的episode（对于某些任务）
            return True
            
        except:
            return True
    
    def _generate_her_for_episode(self, task_id: str):
        """为完成的episode生成HER经验"""
        try:
            episode_rollouts = self.episode_buffer[task_id]
            episode_length = len(episode_rollouts)
            
            if episode_length == 0:
                return
            
            # 判断episode是否成功
            is_successful = self._is_episode_successful(episode_rollouts)
            
            if is_successful:
                self.her_stats['successful_episodes'] += 1
            else:
                self.her_stats['failed_episodes'] += 1
            
            # 更新平均episode长度
            total_episodes = self.her_stats['successful_episodes'] + self.her_stats['failed_episodes']
            self.her_stats['average_episode_length'] = (
                (self.her_stats['average_episode_length'] * (total_episodes - 1) + episode_length) / total_episodes
            )
            
            # 为失败的episode生成HER经验
            if not is_successful:
                her_experiences = self._create_her_experiences(episode_rollouts)
                
                for her_exp in her_experiences:
                    if len(self.her_buffer) < self.her_buffer.maxlen:
                        self.her_buffer.append(her_exp)
                        self.her_stats['total_her_generated'] += 1
            
            # 将完成的episode移动到completed_episodes
            self.completed_episodes.append({
                'task_id': task_id,
                'rollouts': episode_rollouts,
                'length': episode_length,
                'successful': is_successful
            })
            
            # 清理episode buffer
            del self.episode_buffer[task_id]
            
            print(f"Episode {task_id} completed: length={episode_length}, "
                  f"successful={is_successful}, HER generated={len(her_experiences) if not is_successful else 0}")
            
        except Exception as e:
            print(f"HER generation error for episode {task_id}: {e}")
    
    def _is_episode_successful(self, episode_rollouts: List[DataProto]) -> bool:
        """判断episode是否成功"""
        try:
            # 检查最终奖励
            final_rollout = episode_rollouts[-1]
            
            if 'reward' in final_rollout.batch:
                final_reward = final_rollout.batch['reward'][0].item()
                return final_reward > self.her_config.reward_threshold
            
            # 检查准确性
            if 'acc' in final_rollout.batch:
                final_acc = final_rollout.batch['acc'][0].item()
                return final_acc > 0.5
            
            # 默认基于某些指标判断
            return False
            
        except:
            return False
    
    def _create_her_experiences(self, episode_rollouts: List[DataProto]) -> List[DataProto]:
        """创建HER经验"""
        her_experiences = []
        
        try:
            for i, rollout in enumerate(episode_rollouts):
                # 为每个rollout生成k个HER经验
                for _ in range(self.her_config.replay_k):
                    her_exp = self._create_single_her_experience(rollout, episode_rollouts, i)
                    if her_exp is not None:
                        her_experiences.append(her_exp)
            
        except Exception as e:
            print(f"HER experience creation error: {e}")
        
        return her_experiences
    
    def _create_single_her_experience(self, rollout: DataProto, episode_rollouts: List[DataProto], rollout_idx: int) -> Optional[DataProto]:
        """创建单个HER经验"""
        try:
            # 复制原始rollout
            her_rollout = copy.deepcopy(rollout)
            
            # 根据策略选择新目标
            new_goal = self._select_her_goal(episode_rollouts, rollout_idx)
            
            if new_goal is None:
                return None
            
            # 重新标记目标和奖励
            her_rollout = self._relabel_rollout(her_rollout, new_goal)
            
            # 标记为HER经验
            her_rollout.meta_info = her_rollout.meta_info or {}
            her_rollout.meta_info['is_her'] = True
            her_rollout.meta_info['her_strategy'] = self.her_config.strategy.value
            
            return her_rollout
            
        except Exception as e:
            print(f"Single HER experience creation error: {e}")
            return None
    
    def _select_her_goal(self, episode_rollouts: List[DataProto], current_idx: int) -> Optional[Any]:
        """根据HER策略选择新目标"""
        episode_length = len(episode_rollouts)
        
        if self.her_config.strategy == HERStrategy.FINAL:
            # 使用最终状态作为目标
            return self._extract_achieved_goal(episode_rollouts[-1])
        
        elif self.her_config.strategy == HERStrategy.FUTURE:
            # 使用未来状态作为目标
            if current_idx < episode_length - 1:
                future_idx = random.randint(current_idx + 1, episode_length - 1)
                return self._extract_achieved_goal(episode_rollouts[future_idx])
            else:
                return self._extract_achieved_goal(episode_rollouts[-1])
        
        elif self.her_config.strategy == HERStrategy.EPISODE:
            # 使用episode中随机状态作为目标
            random_idx = random.randint(0, episode_length - 1)
            return self._extract_achieved_goal(episode_rollouts[random_idx])
        
        elif self.her_config.strategy == HERStrategy.RANDOM:
            # 使用随机目标
            return self._generate_random_goal()
        
        return None
    
    def _extract_achieved_goal(self, rollout: DataProto) -> Optional[Any]:
        """从rollout中提取达成的目标"""
        try:
            # 尝试从不同可能的位置提取目标
            if self.her_config.achieved_goal_key in rollout.batch:
                return rollout.batch[self.her_config.achieved_goal_key][0]
            
            # 如果没有明确的achieved_goal，可以从输出中推断
            if 'output' in rollout.non_tensor_batch:
                output_text = rollout.non_tensor_batch['output'][0]
                return self._extract_goal_from_text(output_text)
            
            # 其他启发式方法
            return None
            
        except:
            return None
    
    def _extract_goal_from_text(self, text: str) -> Optional[str]:
        """从文本输出中提取目标（启发式方法）"""
        try:
            # 这里可以根据具体任务定制
            # 例如，对于代码生成任务，可以提取函数名或关键词
            # 对于数学问题，可以提取最终答案
            
            # 简单示例：提取最后一个句子或关键词
            sentences = text.strip().split('.')
            if sentences:
                return sentences[-1].strip()
            
            return text.strip()[:50]  # 截取前50个字符作为目标
            
        except:
            return None
    
    def _generate_random_goal(self) -> str:
        """生成随机目标"""
        # 可以根据任务类型生成随机目标
        random_goals = [
            "solve the problem correctly",
            "provide accurate answer",
            "complete the task",
            "generate valid output"
        ]
        return random.choice(random_goals)
    
    def _relabel_rollout(self, rollout: DataProto, new_goal: Any) -> DataProto:
        """重新标记rollout的目标和奖励"""
        try:
            # 更新目标
            if self.her_config.goal_key in rollout.batch:
                rollout.batch[self.her_config.goal_key][0] = new_goal
            elif self.her_config.goal_key in rollout.non_tensor_batch:
                rollout.non_tensor_batch[self.her_config.goal_key][0] = new_goal
            
            # 重新计算奖励
            new_reward = self._compute_her_reward(rollout, new_goal)
            
            if 'reward' in rollout.batch:
                rollout.batch['reward'][0] = new_reward
            
            # 更新其他相关字段
            if 'acc' in rollout.batch:
                rollout.batch['acc'][0] = 1.0 if new_reward > 0 else 0.0
            
            return rollout
            
        except Exception as e:
            print(f"Rollout relabeling error: {e}")
            return rollout
    
    def _compute_her_reward(self, rollout: DataProto, goal: Any) -> float:
        """为HER经验计算新奖励"""
        try:
            # 提取当前达成的目标
            achieved = self._extract_achieved_goal(rollout)
            
            if achieved is None:
                return 0.0
            
            # 计算目标匹配度
            if isinstance(goal, str) and isinstance(achieved, str):
                # 文本相似度比较
                similarity = self._compute_text_similarity(goal, achieved)
                return 1.0 if similarity > 0.8 else 0.0
            
            # 其他类型的目标比较
            if goal == achieved:
                return 1.0
            
            return 0.0
            
        except:
            return 0.0
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        try:
            # 简单的词汇重叠相似度
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except:
            return 0.0
        
    def get_her_stats(self) -> Dict:
        """获取HER统计信息"""
        return {
            **self.her_stats,
            'original_buffer_size': len(self.original_buffer),
            'her_buffer_size': len(self.her_buffer),
            'total_buffer_size': self._total_size(),
            'active_episodes': len(self.episode_buffer),
            'completed_episodes': len(self.completed_episodes),
            'her_enabled': self.her_config.enabled,
            'her_strategy': self.her_config.strategy.value,
            'replay_k': self.her_config.replay_k
        }
    
    def print_her_summary(self):
        """打印HER统计摘要"""
        stats = self.get_her_stats()
        print("\n=== HER Buffer Summary ===")
        print(f"Strategy: {stats['her_strategy']}")
        print(f"Total experiences: {stats['total_buffer_size']}")
        print(f"  - Original: {stats['original_buffer_size']}")
        print(f"  - HER: {stats['her_buffer_size']}")
        print(f"Episodes: {stats['successful_episodes']} successful, {stats['failed_episodes']} failed")
        print(f"Average episode length: {stats['average_episode_length']:.1f}")
        print(f"HER sample ratio: {stats['her_sample_ratio']:.2%}")
        print(f"Total HER generated: {stats['total_her_generated']}")
        print("========================\n")