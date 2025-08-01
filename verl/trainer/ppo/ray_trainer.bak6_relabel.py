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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import statistics
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pprint import pprint
from typing import Callable, Type, Tuple, Union
import uuid
from omegaconf import OmegaConf, open_dict
import numpy as np
from codetiming import Timer
import random
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.dataset.rob_dataset import BufferedDataLoader

# from verl.trainer.ppo.async_data_generator import AsyncDataGenerator
# from verl.trainer.ppo.async_experience_buffer import AsyncExperienceBuffer, AsyncExperienceBufferWithHER

WorkerType = Type[Worker]

import asyncio
import threading
import time
from collections import defaultdict, deque
from typing import Optional, List, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import copy
import numpy as np
from verl import DataProto

class SafeExecutorWrapper:
    """安全的执行器包装器，处理StopIteration异常"""
    
    def __init__(self, executor):
        self.executor = executor
    
    async def run_safe(self, loop, func, *args, **kwargs):
        """安全运行函数，处理StopIteration"""
        try:
            result = await loop.run_in_executor(
                self.executor,
                self._safe_wrapper,
                func,
                args,
                kwargs
            )
            return result
        except RuntimeError as e:
            if "StopIteration" in str(e):
                return None
            raise e
    
    def _safe_wrapper(self, func, args, kwargs):
        """安全包装器函数"""
        try:
            return func(*args, **kwargs)
        except StopIteration:
            return None  # 转换StopIteration为None
        except Exception as e:
            raise e

class AsyncDataGenerator:
    def __init__(self, trainer, actor_rollout_wg, tokenizer, train_dataloader, config):
        self.trainer = trainer
        self.actor_rollout_wg = actor_rollout_wg
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.config = config
        
        # 关键参数
        self.n_samples = config.data.n_samples  # 4
        self.batch_size = self.actor_rollout_wg.world_size #config.data.train_batch_size  # 32
        
        # 获取reward_fn和filter函数的引用
        self.reward_fn = trainer.reward_fn
        self.filter_func = trainer.filter
        
        # 其他初始化...
        self.safe_executor = SafeExecutorWrapper(
            ThreadPoolExecutor(max_workers=7, thread_name_prefix="dataloader")
        )
        self.generation_thread = None
        self.stop_event = threading.Event()
        self.generation_metrics = defaultdict(list)
        self.total_generated = 0
        self.total_filtered = 0
        self.traffic_sign = False
        
    def start_generation(self):
        """启动异步数据生成线程"""
        self.generation_thread = threading.Thread(
            target=self._run_async_generation,
            daemon=True
        )
        self.generation_thread.start()
        print("Async data generation started!")
        
    def stop_generation(self):
        """停止数据生成"""
        self.stop_event.set()
        if self.generation_thread:
            self.generation_thread.join()
        print("Async data generation stopped!")
    
    def _run_async_generation(self):
        """在独立线程中运行异步生成循环"""
        # 创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # try:
        loop.run_until_complete(self._async_generation_loop())
        # finally:
        #     loop.close()
            
    def _safe_get_next_batch(self) -> Optional[dict]:
        """安全的获取下一个批次，避免StopIteration泄露"""
        try:
            return self.train_dataloader.get_next_batch()
        except StopIteration:
            return None  # 将StopIteration转换为None
        except Exception as e:
            # 其他异常继续抛出
            raise e
    
    async def _async_generation_loop(self):
        """异步生成主循环 - 持续生成模式"""
        generation_round = 0
        
        while not self.stop_event.is_set():
            try:
                generation_round += 1
                print(f"Starting generation epoch {generation_round}")
                self.train_dataloader.start_new_epoch()
                
                # 持续生成数据
                await self._generate_batch_async()
                
                # 短暂休息，避免过度占用资源
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                print("Generation loop cancelled")
                break
            except Exception as e:
                print(f"Generation loop error: {e}")
                await asyncio.sleep(1)
                
    
    async def _get_next_batch_async(self) -> Optional[dict]:
        """异步获取下一个批次"""
        loop = asyncio.get_event_loop()
        
        try:
            # 在线程池中执行同步调用，并处理StopIteration
            with concurrent.futures.ThreadPoolExecutor() as executor:
                batch_dict = await loop.run_in_executor(
                    executor,
                    self._safe_get_next_batch  # 使用安全的包装函数
                )
            return batch_dict
        except RuntimeError as e:
            if "StopIteration" in str(e):
                return None  # 数据耗尽
            raise e
    
    async def _process_batch_async(self, batch_dict: dict, buffer_batch: List, n_samples: int) -> Tuple[Optional[DataProto], dict]:
        """异步处理单个批次"""
        
        try:
            with Timer(name='gen', text="{name}: {seconds:.1f} seconds") as timer:
                # 创建新批次
                newbatch: DataProto = DataProto.from_single_dict(batch_dict)
                
                # 合并buffer数据
                if len(buffer_batch) > 0:
                    newbatch = DataProto.concat([buffer_batch, newbatch])
                
                # 选择生成批次的键
                gen_batch = newbatch.select(
                    batch_keys=['task_id', 'trial_id'],
                    non_tensor_batch_keys={"task_suite_name"},
                    meta_info_keys={}
                )
                
                # 添加唯一ID
                newbatch.non_tensor_batch['uid'] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(newbatch.batch))],
                    dtype=object
                )
                
                # 创建批次列表
                batch_lst = sum([
                    [newbatch[i:i + 1] for _ in range(n_samples)] 
                    for i in range(len(newbatch))
                ], [])
                
                # 设置生成元信息
                gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'n_samples': n_samples,
                    'pad_token_id': self.tokenizer.pad_token_id,
                }
                
                # 异步生成序列
                gen_batch_output = await self._generate_sequences_async(gen_batch)
                
                # 合并结果
                roll_batch = DataProto.concat(batch_lst)
                roll_batch = roll_batch.union(gen_batch_output)
            
            # 计算指标
            batch_metrics = {
                'timing/gen': timer.last,
                'timing/verify': 0,  # 后续验证步骤的时间
                'timing/acc&trunc_filter': 0,
                'timing/filter_format_error': 0,
                'timing/compute_all_entropy': 0,
            }
            
            return roll_batch, batch_metrics
            
        except Exception as e:
            print(f"Process batch error: {e}")
            return None, {}
    
    async def _generate_sequences_async(self, gen_batch: DataProto) -> DataProto:
        """异步生成序列"""
        loop = asyncio.get_event_loop()
        
        # 使用模型锁保护生成过程
        # with self.trainer.model_lock:
        # 在线程池中执行生成
        with concurrent.futures.ThreadPoolExecutor() as executor:
            gen_batch_output = await loop.run_in_executor(
                executor,
                self.actor_rollout_wg.generate_sequences,
                gen_batch
            )
        
        return gen_batch_output
    
    def get_generation_stats(self) -> dict:
        """获取包含过滤统计的生成信息"""
        base_stats = {
            'total_generated': self.total_generated,
            'is_running': self.generation_thread is not None and self.generation_thread.is_alive()
        }
        
        # 计算过滤相关统计
        filter_pass_rates = self.generation_metrics.get('filter_pass_rate', [])
        if filter_pass_rates:
            base_stats.update({
                'average_filter_pass_rate': np.mean(filter_pass_rates),
                'min_filter_pass_rate': np.min(filter_pass_rates),
                'max_filter_pass_rate': np.max(filter_pass_rates),
                'filter_pass_rate_std': np.std(filter_pass_rates)
            })
        
        # 计算时间统计
        for timing_key in ['timing/gen', 'timing/verify', 'timing/acc&trunc_filter']:
            times = self.generation_metrics.get(timing_key, [])
            if times:
                base_stats[f'average_{timing_key}'] = np.mean(times)
                base_stats[f'total_{timing_key}'] = np.sum(times)
                
        for score_key in ['train_verify_score/all', 'format_score/all', 'train_verify_score_wo_format/all']:
            times = self.generation_metrics.get(score_key, [])
            if times:
                base_stats[score_key] = np.mean(times)
        
        return base_stats
        
    async def _get_from_buffer_async(self, batch_size: int, world_size: int) -> List:
        """异步从dataloader buffer获取数据"""
        loop = asyncio.get_event_loop()
        
        # 在线程池中执行同步调用
        with concurrent.futures.ThreadPoolExecutor() as executor:
            buffer_batch = await loop.run_in_executor(
                executor,
                self.train_dataloader.get_from_buffer,
                batch_size,
                world_size
            )
        
        return buffer_batch
    
    async def _generate_batch_async(self) -> Optional[DataProto]:
        """异步生成数据 - 改为增量添加到buffer"""
        try:
            # 不再等待完整批次，而是持续生成并添加
            generation_count = 0
            target_generations = len(self.train_dataloader) #self.config.data.train_batch_size # // self.actor_rollout_wg.world_size  # 每轮生成多少个原始样本
            
            buffer_batch = []
            if self.train_dataloader.buffer_size() > 0:
                buffer_batch = await self._get_from_buffer_async(
                    target_generations, self.actor_rollout_wg.world_size
                )
            
            # 准备指标收集
            all_metrics = defaultdict(list)
            
            while generation_count < target_generations:
                # 检查buffer是否接近满
                if (hasattr(self.trainer, 'experience_buffer') and 
                    self.trainer.experience_buffer.size() >= self.trainer.config.data.max_buffer_size * 0.9):
                    print("Buffer nearly full, pausing generation...")
                    await asyncio.sleep(1)
                    continue
                
                while self.traffic_sign:
                    time.sleep(10)
                
                # 获取下一个原始批次数据
                batch_dict = await self._get_next_batch_async()
                if batch_dict is None:
                    print(f"No more data available from dataloader, target_generations: {generation_count}/{target_generations}")
                    break
                
                # 处理单个批次（包含生成、验证、过滤）
                processed_batch, batch_metrics = await self._process_single_batch_with_nsamples(
                    batch_dict, buffer_batch
                )
                
                if processed_batch is not None and len(processed_batch.batch) > 0:
                    # 立即添加到buffer，不等待完整d批次
                    if hasattr(self.trainer, 'experience_buffer'):
                        self.trainer.experience_buffer.add_batch(processed_batch)
                        # print(f"Immediately added {len(processed_batch.batch)} rollouts to buffer, "
                        #       f"buffer size now: {self.trainer.experience_buffer.size()}")
                    
                    # 收集指标
                    for key, value in batch_metrics.items():
                        if isinstance(value, (list, tuple)):
                            all_metrics[key].extend(value)
                        else:
                            all_metrics[key].append(value)
                    
                    # 清空buffer_batch（只在第一次使用）
                    if len(buffer_batch) > 0:
                        buffer_batch = []
                    
                    generation_count += 1
                    self.total_generated += len(processed_batch.batch)
                    
                    print(f"Generated and added batch {generation_count}/{target_generations}, "
                          f"Buffer size: {self.trainer.experience_buffer.size()}/{self.config.data.max_buffer_size}")
                    
                else:
                    print(f"Batch processing failed or resulted in empty batch")
                    # 继续尝试下一个批次
                    generation_count += 1
            
                # 更新全局指标
                for key, values in all_metrics.items():
                    self.generation_metrics[key].extend(values if isinstance(values, list) else [values])
            
            # 返回None，因为数据已经直接添加到buffer了
            return None
            
        except Exception as e:
            print(f"Generate batch with incremental addition error: {e}")
            return None
    
    async def _process_single_batch_with_nsamples(self, batch_dict: dict, buffer_batch: List) -> Tuple[Optional[DataProto], dict]:
        """处理单个批次，包含生成、验证和过滤的完整流程"""
        start_time = time.time()
        
        try:
            # 1. 生成rollout数据（原有逻辑）
            roll_batch, gen_metrics = await self._generate_rollouts(batch_dict, buffer_batch)
        
            if roll_batch is None:
                return None, {}
            
            print(f"Generated {len(roll_batch.batch)} rollouts")
            
            # 2. 验证阶段
            verified_batch, verify_metrics = await self._verify_batch_async(roll_batch)
            
            if verified_batch is None:
                return None, gen_metrics
            
            # 3. 过滤阶段
            filtered_batch, filter_metrics = await self._filter_batch_async(verified_batch)
            
            if filtered_batch is None or len(filtered_batch.batch) == 0:
                print("No rollouts passed filtering")
                # 根据filter_warmup策略决定是否使用原始数据
                if self.config.data.filter_warmup:
                    # filter_warmup模式下，如果过滤后为空，使用原始数据
                    final_batch = verified_batch if len(verified_batch.batch) > 0 else roll_batch
                else:
                    # 正常模式下，只使用过滤后的数据
                    final_batch = filtered_batch
            else:
                final_batch = filtered_batch
            
            # 合并所有指标
            combined_metrics = {**gen_metrics, **verify_metrics, **filter_metrics}
            combined_metrics['rollouts_before_filter'] = len(roll_batch.batch)
            combined_metrics['rollouts_after_filter'] = len(final_batch.batch)
            combined_metrics['filter_pass_rate'] = len(final_batch.batch) / len(roll_batch.batch) if len(roll_batch.batch) > 0 else 0
            
            print(f"Final batch: {len(final_batch.batch)} rollouts "
                  f"(filter pass rate: {combined_metrics['filter_pass_rate']:.2%})")
            
            return final_batch, combined_metrics
            
        except Exception as e:
            print(f"Process batch with verification and filtering error: {e}")
            return None, {}
    
    async def _generate_rollouts(self, batch_dict: dict, buffer_batch: List) -> Tuple[Optional[DataProto], dict]:
        """生成rollout数据的原始逻辑"""
        
        try:
            with Timer(name='gen', text="{name}: {seconds:.1f} seconds") as timer:
                # 创建新批次
                newbatch: DataProto = DataProto.from_single_dict(batch_dict)
                
                # 合并buffer数据（如果有）
                if len(buffer_batch) > 0:
                    newbatch = DataProto.concat([buffer_batch, newbatch])
                
                # 准备生成批次
                gen_batch = newbatch.select(
                    batch_keys=['task_id', 'trial_id'],
                    non_tensor_batch_keys={"task_suite_name"},
                    meta_info_keys={}
                )
                
                # 为每个样本添加唯一ID
                newbatch.non_tensor_batch['uid'] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(newbatch.batch))],
                    dtype=object
                )
                
                # 为每个原始样本创建n_samples个副本
                batch_lst = []
                for i in range(len(newbatch)):
                    single_sample = newbatch[i:i + 1]
                    for _ in range(self.n_samples):
                        batch_lst.append(single_sample)
                
                # 设置生成元信息
                gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'n_samples': self.n_samples,
                    'pad_token_id': self.tokenizer.pad_token_id,
                }
                
                # 异步生成序列
                gen_batch_output = await self._generate_sequences_async(gen_batch)
                
                # 合并结果
                roll_batch = DataProto.concat(batch_lst)
                roll_batch = roll_batch.union(gen_batch_output)
            
            # 计算生成指标
            gen_metrics = {
                'timing/gen': timer.last,
                'rollouts_generated': len(roll_batch.batch),
                'original_samples': len(newbatch),
                'n_samples_used': self.n_samples
            }
            
            return roll_batch, gen_metrics
            
        except Exception as e:
            print(f"Generate rollouts error: {e}")
            return None, {}
        
    async def _verify_batch_async(self, roll_batch: DataProto) -> Tuple[Optional[DataProto], dict]:
        """异步验证批次 - 包含准确性和格式验证"""
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            
            # 在线程池中执行验证
            verify_result = await loop.run_in_executor(
                self.safe_executor.executor,
                self._sync_verify_batch,
                roll_batch
            )
            
            if verify_result is None:
                return None, {}
            
            scores_tensor, reward_metrics, format_metrics, reward_format_metrics = verify_result
            
            # 收集验证指标
            verify_metrics = {
                'timing/verify': time.time() - start_time
            }
            
            # 添加各种评分指标
            for k, v in reward_metrics.items():
                verify_metrics[f'train_verify_score/{k}'] = v
                
            for k, v in format_metrics.items():
                verify_metrics[f'format_score/{k}'] = v
                
            for k, v in reward_format_metrics.items():
                verify_metrics[f'train_verify_score_wo_format/{k}'] = v
            
            print(f"Verification completed: {len(roll_batch.batch)} rollouts verified")
            
            return roll_batch, verify_metrics
            
        except Exception as e:
            print(f"Verify batch async error: {e}")
            return None, {}
    
    def _sync_verify_batch(self, roll_batch: DataProto):
        """同步验证方法（在线程池中执行）"""
        try:
            return self.reward_fn.verify(roll_batch)
        except Exception as e:
            print(f"Sync verify error: {e}")
            return None
        
    async def _filter_batch_async(self, roll_batch: DataProto) -> Tuple[Optional[DataProto], dict]:
        """异步过滤批次 - 包含准确性和截断过滤"""
        start_time = time.time()
        
        try:
            # 检查是否需要过滤
            if not (self.config.data.filter_accuracy or self.config.data.filter_truncated):
                # 不需要过滤，直接返回
                return roll_batch, {'timing/acc&trunc_filter': 0, 'filter_applied': False}
            
            print(f"before filtering: {len(roll_batch)}")
            
            loop = asyncio.get_event_loop()
            
            # 在线程池中执行过滤
            filtered_roll_batch = await loop.run_in_executor(
                self.safe_executor.executor,
                self._sync_filter_batch,
                roll_batch
            )
            
            filter_time = time.time() - start_time
            
            if filtered_roll_batch is not None:
                print(f"after filtering: {len(filtered_roll_batch)}")
                
                filter_metrics = {
                    'timing/acc&trunc_filter': filter_time,
                    'filter_applied': True,
                    'filtered_rollouts': len(filtered_roll_batch.batch),
                    'filter_ratio': len(filtered_roll_batch.batch) / len(roll_batch.batch) if len(roll_batch.batch) > 0 else 0
                }
                
                return filtered_roll_batch, filter_metrics
            else:
                # 过滤失败，返回原始数据
                print("Filtering failed, returning original batch")
                return roll_batch, {
                    'timing/acc&trunc_filter': filter_time,
                    'filter_applied': False,
                    'filter_failed': True
                }
            
        except Exception as e:
            print(f"Filter batch async error: {e}")
            # 过滤出错，返回原始数据
            return roll_batch, {
                'timing/acc&trunc_filter': time.time() - start_time,
                'filter_applied': False,
                'filter_error': str(e)
            }
    
    def _sync_filter_batch(self, roll_batch: DataProto) -> Optional[DataProto]:
        """同步过滤方法（在线程池中执行）"""
        try:
            # 调用原有的过滤函数
            # 注意：这里假设roll_batch.batch['acc']存在并且是正确的格式
            filtered_roll_batch = self.filter_func(
                roll_batch.batch['acc'].unsqueeze(1), 
                roll_batch, 
                self.n_samples
            )
            
            return filtered_roll_batch
            
        except Exception as e:
            print(f"Sync filter error: {e}")
            return None
        
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

class TimestampedRollout:
    """带时间戳的rollout包装器"""
    def __init__(self, rollout: DataProto, timestamp: float = None, generation_id: int = None):
        self.rollout = rollout
        self.timestamp = timestamp or time.time()
        self.generation_id = generation_id or 0  # 生成批次ID
        
    def __lt__(self, other):
        """用于堆排序，较早的时间戳优先"""
        return self.timestamp < other.timestamp

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
        
        # 重要性采样参数
        self.current_policy_version = 0
        self.importance_sampling_beta = getattr(config.data, 'importance_sampling_beta', 0.4)
        self.priority_alpha = getattr(config.data, 'priority_alpha', 0.6)
        self.max_version_diff = getattr(config.data, 'max_version_diff', 2)
        self.quality_threshold = getattr(config.data, 'quality_threshold', 0.3)
        
    def set_policy_version(self, version):
        """更新当前policy版本"""
        self.current_policy_version = version
    
    def add_batch(self, batch_data: DataProto):
        """添加批次数据到缓冲区 - 将批次拆分为单个rollout"""
        policy_version = self.current_policy_version
        
        with self.not_full:
            # 将批次拆分为单个rollout存储
            for i in range(len(batch_data.batch)):
                while len(self.buffer) >= self.config.data.max_buffer_size:
                    self._remove_oldest_from_buffer(self.buffer, self.config.data.min_buffer_size)
                    self.not_full.wait()  # 等待空间
                    
                # while len(self.buffer) > 2 * self.config.data.min_buffer_size:
                #     self._remove_oldest_from_buffer(self.buffer, self.config.data.min_buffer_size)
                #     self.not_full.wait()  # 等待空间
                
                # 提取单个rollout
                single_rollout = batch_data[i:i+1]
                rollout_with_meta = {
                    'data': single_rollout,
                    'policy_version': policy_version,
                    'timestamp': time.time(),
                    'quality_score': self._calculate_quality_score(single_rollout)
                }
                self.buffer.append(rollout_with_meta)
            
            self._cleanup_outdated_data()
            
            print(f"Added {len(batch_data.batch)} rollouts to buffer, "
                  f"buffer size: {len(self.buffer)}")
            
            self.not_empty.notify_all()  # 通知有新数据
            
    def _calculate_quality_score(self, rollout):
        """计算rollout的质量分数"""
        # 基于reward计算质量分数
        if 'complete' in rollout.batch:
            return float(rollout.batch['complete'].tolist()[0])
        # if hasattr(rollout, 'complete') and rollout.rewards:
        #     return float(np.mean(rollout.rewards))
        # elif hasattr(rollout.batch[0], 'rewards'):
        #     rewards = rollout.batch[0]['rewards']
        #     if isinstance(rewards, torch.Tensor):
        #         return float(rewards.mean().item())
        #     else:
        #         return float(np.mean(rewards))
        # return 0.0
        
    def _cleanup_outdated_data(self):
        """清理过时的数据"""
        if not self.buffer:
            return
            
        valid_buffer = deque(maxlen=self.config.data.max_buffer_size)
        removed_count = 0
        
        for item in self.buffer:
            version_diff = self.current_policy_version - item['policy_version']
            if version_diff <= self.max_version_diff: # and item['quality_score'] >= self.quality_threshold:
                valid_buffer.append(item)
            else:
                removed_count += 1
        
        if removed_count > 0:
            self.buffer = valid_buffer
            print(f"Removed {removed_count} outdated/low-quality rollouts")
            
    def _compute_sampling_probabilities(self, sampling_strategy='adv_priority', batch_id2mean=[], batch_id2std=[], advantages=[], version_diffs=[]):
        """
        计算所有buffer中数据的采样概率
        
        Returns:
            sampling_probs: torch.Tensor, shape (buffer_size,)
        """
        if not self.buffer:
            return torch.tensor([])
        
        if sampling_strategy == 'adv_priority':
            # 基于advantage计算优先级
            scores = torch.tensor(advantages, dtype=torch.float32)
            priority_weights = torch.pow(scores, self.priority_alpha)
            sampling_probs = priority_weights / torch.sum(priority_weights) # (bs * sample)
        
        elif sampling_strategy == 'mean_reward_priority':
            # 基于质量分数和新鲜度计算优先级
            scores = []
            for id2mean, version_diff in zip(batch_id2mean, version_diffs):
                freshness_score = 1.0 / (1.0 + 0.1 * version_diff)
                
                # 组合分数：70%质量 + 30%新鲜度
                combined_score = 0.5 * id2mean + 0.5 * freshness_score
                scores.append(combined_score)
            
            # 转换为采样概率
            scores = torch.tensor(scores, dtype=torch.float32)
            priority_weights = torch.pow(scores, self.priority_alpha)
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
            priority_weights = torch.pow(scores, self.priority_alpha)
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
            
        else:  # mixed or other
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
        
        return sampling_probs
    
    def _compute_importance_weights(self, sampled_indices, sampling_probs):
        """
        使用标准重要性采样公式计算权重
        
        公式: w_i = (N * P(i))^(-beta) / max_i(w_i)
        
        Args:
            sampled_indices: 采样的索引列表
            sampling_probs: 所有样本的采样概率 tensor
            
        Returns:
            importance_weights: torch.Tensor, 归一化的重要性权重
        """
        N = int(len(self.buffer) / self.n_samples)  # 总样本数
        
        new_sampled_indices = []
        for sampled_indice in sampled_indices:
            new_sampled_indices.append(int(sampled_indice//self.n_samples))
        
        # 获取采样样本的概率
        selected_probs = sampling_probs[new_sampled_indices]
        
        # import ipdb;ipdb.set_trace()
        
        # 使用标准重要性采样公式: w_i = (N * P(i))^(-beta)
        raw_weights = torch.pow(N * selected_probs, -self.importance_sampling_beta)
        
        # 归一化：除以最大权重
        max_weight = torch.max(raw_weights)
        if max_weight > 0:
            normalized_weights = raw_weights / max_weight
        else:
            normalized_weights = torch.ones_like(raw_weights)
            
        new_normalized_weights = []
        for i, normalized_weight in enumerate(normalized_weights):
            for samples in range(i, i + self.n_samples):
                new_normalized_weights.append(normalized_weights[i])
        
        return torch.tensor(new_normalized_weights)
    
    def _sample_indices_by_probability(self, indices, sampling_probs, num_samples):
        """
        根据概率采样索引
        
        Args:
            sampling_probs: 采样概率
            num_samples: 需要采样的数量
            
        Returns:
            sampled_indices: 采样的索引
        """
        buffer_size = len(self.buffer)
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
    
    def sample_batch_with_important_sampling(self, batch_size: int = None, sampling_strategy='adv_priority', epsilon=1e-4) -> Optional:
        """
        改进的采样方法，使用正确的重要性权重计算
        """
        if batch_size is None:
            batch_size = self.train_batch_size
        
        total_rollouts_needed = batch_size * self.n_samples
        
        with self.not_empty:
            while len(self.buffer) < total_rollouts_needed:
                if not self.not_empty.wait(timeout=1.0):
                    return None
                
            batch_indexs = [idx for idx in range(0, len(self.buffer), self.n_samples)]
            uid_indexs = [self.buffer[i]["data"].non_tensor_batch['uid'].tolist()[0] for i in range(len(self.buffer))]
            quality_scores = [self.buffer[i]["quality_score"] for i in range(len(self.buffer))]
            advantages = [self.buffer[i]["quality_score"] * 5.0 for i in range(len(self.buffer))]
            id2score, id2mean, id2std, id2diffs = [], [], [], []
            
            # print("batch_indexs",batch_indexs, len(self.buffer), len(batch_indexs))
            # print("quality_scores",quality_scores, len(self.buffer), len(quality_scores))
            
            for i in batch_indexs:
                tmp = []
                tmp_diff = 0
                for samples in range(i, i + self.n_samples):
                    tmp.append(quality_scores[samples])
                    tmp_diff = self.current_policy_version - self.buffer[samples]['policy_version']
                id2score.append(tmp)
                id2diffs.append(tmp_diff)
            
            for idx in range(len(id2score)):
                if len(id2score[idx]) == 1:
                    id2mean.append(torch.tensor(0.0))
                    id2std.append(torch.tensor(1.0))
                elif len(id2score[idx]) > 1:
                    id2mean.append(torch.mean(torch.tensor(id2score[idx])))
                    id2std.append(torch.std(torch.tensor([id2score[idx]])))
                
            for i in batch_indexs:
                for samples in range(i, i + self.n_samples):
                    advantages[samples] = (advantages[samples] - id2mean[int(i//self.n_samples)]) / (id2std[int(i//self.n_samples)] + epsilon)
            
            # print("id2score",id2score)
            # print("advantages",advantages)
            print("id2diffs",id2diffs)
            # print("id2mean",id2mean)
            # print("id2std",id2std)
            self._relabel_rollout(batch_indexs, id2mean)
            # 1. 计算所有样本的采样概率
            sampling_probs = self._compute_sampling_probabilities(sampling_strategy, id2mean, id2std, advantages, id2diffs)
            print("sampling_probs",sampling_probs)
            
            if len(sampling_probs) == 0:
                return None
            
            # 2. 根据概率采样索引
            sampled_indices = self._sample_indices_by_probability(
                batch_indexs, sampling_probs, int(total_rollouts_needed / self.n_samples)
            )
            # print("sampled_indices",sampled_indices)
            
            # 3. 计算重要性权重
            importance_weights = self._compute_importance_weights(
                sampled_indices, sampling_probs
            )
            print("importance_weights",importance_weights)
            
            # 4. 提取采样的数据
            sampled_items = []
            for i in sampled_indices:
                for samples in range(i, i + self.n_samples):
                    sampled_items.append(self.buffer[samples])
            # sampled_items = [self.buffer[i] for i in sampled_indices]
            sampled_rollouts = [item['data'] for item in sampled_items]
            
            # 5. 构建训练批次
            if sampled_rollouts:
                # 合并成训练批次
                training_batch = DataProto.concat(sampled_rollouts)
                print(f"Sampled {len(training_batch.batch)} rollouts for training "
                      f"(batch_size={batch_size}, n_samples={self.n_samples}, "
                      f"total={total_rollouts_needed})")
            
            # 6. 添加重要性权重到批次中
            training_batch.batch["importance_weights"] = importance_weights
            
            # 7. 计算和打印统计信息
            self._log_sampling_stats(importance_weights, sampling_strategy, sampled_indices)
            
            self.not_full.notify_all()
            return training_batch
        
    def _relabel_rollout(self, batch_indexs, id2mean):
        for i in batch_indexs:
            for samples in range(i, i + self.n_samples):
                if not self.buffer[samples]["data"].batch["complete"]:
                    self.buffer[samples]["data"].batch["complete"] = torch.tensor([id2mean[int(i//self.n_samples)]])
                    self.buffer[samples]["data"].batch['finish_step'] = torch.tensor([random.randint(256, 511)])
        
    def sample_batch_grouped_with_important_sampling(self, batch_size: int = None) -> Optional[List]:
        """采样分组的批次，保持重要性权重"""
        if batch_size is None:
            batch_size = self.train_batch_size
        
        total_rollouts_needed = batch_size * self.n_samples
        
        with self.not_empty:
            while len(self.buffer) < total_rollouts_needed:
                if not self.not_empty.wait(timeout=1.0):
                    return None
            
            # 使用相同的采样逻辑
            sampling_probs = self._compute_sampling_probabilities('priority')
            sampled_indices = self._sample_indices_by_probability(
                sampling_probs, total_rollouts_needed
            )
            importance_weights = self._compute_importance_weights(
                sampled_indices, sampling_probs
            )
            
            # 分组返回
            sampled_items = [self.buffer[i] for i in sampled_indices]
            grouped_rollouts = []
            grouped_weights = []
            
            for i in range(0, len(sampled_items), self.n_samples):
                group_items = sampled_items[i:i + self.n_samples]
                group_weights = importance_weights[i:i + self.n_samples]
                
                if len(group_items) == self.n_samples:
                    group_data = [item['data'] for item in group_items]
                    group_data.batch["importance_weights"] = group_weights
                    grouped_rollouts.append(group_data)
                    grouped_weights.append(group_weights)
            
            print(f"Sampled {len(grouped_rollouts)} groups of {self.n_samples} rollouts each")
            self.not_full.notify_all()
            
            # 返回分组数据和对应权重
            return list(zip(grouped_rollouts, grouped_weights))
        
    def _log_sampling_stats(self, importance_weights, sampling_strategy, sampled_indices):
        """记录采样统计信息"""
        weights_np = importance_weights.numpy()
        
        # 基本统计
        avg_weight = np.mean(weights_np)
        std_weight = np.std(weights_np)
        min_weight = np.min(weights_np)
        max_weight = np.max(weights_np)
        
        # 有效样本大小
        effective_sample_size = (np.sum(weights_np) ** 2) / np.sum(weights_np ** 2)
        effective_ratio = effective_sample_size / len(weights_np)
        
        # 版本分布统计
        sampled_versions = [self.buffer[i]['policy_version'] for i in sampled_indices]
        avg_version_diff = self.current_policy_version - np.mean(sampled_versions)
        
        print(f"Sampling stats [{sampling_strategy}]:")
        print(f"  Importance weights: avg={avg_weight:.4f}, std={std_weight:.4f}, range=[{min_weight:.4f}, {max_weight:.4f}]")
        print(f"  Effective sample size: {effective_sample_size:.1f}/{len(weights_np)} (ratio: {effective_ratio:.4f})")
        print(f"  Average version diff: {avg_version_diff:.1f}")
        
        # 警告检查
        if effective_ratio < 0.5:
            print(f"  WARNING: Low effective sample ratio ({effective_ratio:.4f}). Consider reducing beta or adjusting strategy.")
        if std_weight / avg_weight > 0.5:
            print(f"  WARNING: High weight variance. Consider reducing beta parameter.")
    
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
            # sampled_rollouts = self._uniform_sample(total_rollouts_needed)
            sampled_rollouts = self._last_batch_sample(total_rollouts_needed)
            
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
    
    def _last_batch_sample(self, batch_size: int) -> List[DataProto]:
        """均匀采样rollout"""
        available_size = min(batch_size, len(self.buffer))
        # indices = random.sample(range(len(self.buffer)), available_size)
        indices = [-i for i in range(available_size, 0, -1)]
        return [self.buffer[i] for i in indices]
    
    def _remove_oldest_from_buffer(self, buffer, count: int):
        """从指定缓冲区删除最旧的rollout"""
        if len(buffer) <= count:
            buffer.clear()
            return
        
        removed_count = min(count, len(buffer))
        
        for i in range(removed_count):
            buffer.popleft()
        # del buffer[:removed_count] # buffer.slice(start=0, length=removed_count)
        
        print(f"Removed {removed_count} oldest rollouts from buffer")
        
    def clear_buffer(self):
        self.buffer.clear()
    
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
    
class HERRewardStrategy(Enum):
    """HER奖励策略"""
    INDIVIDUAL = "individual"     # 基于单个rollout的奖励
    BATCH_AVERAGE = "batch_avg"   # 基于batch平均奖励
    BATCH_MAJORITY = "batch_maj"  # 基于batch中成功rollout的比例
    ADAPTIVE = "adaptive"         # 自适应策略

@dataclass
class HERConfig:
    """HER配置"""
    enabled: bool = True
    strategy: HERStrategy = HERStrategy.FUTURE
    reward_strategy: HERRewardStrategy = HERRewardStrategy.BATCH_AVERAGE
    replay_k: int = 4  # 每个原始经验生成k个HER经验
    # Batch级别配置
    batch_success_threshold: float = 0.3    # batch平均奖励超过此值算成功
    batch_relabel_threshold: float = 0.7    # 平均奖励超过此值时进行relabel
    min_batch_size_for_her: int = 8         # 进行HER的最小batch大小
    # 传统配置
    future_p: float = 1.0  # future策略中选择未来状态的概率
    reward_threshold: float = 0.1  # 低于此奖励的经验被认为是"失败"的
    max_episode_length: int = 512  # 最大episode长度
    goal_key: str = "goal"  # 目标在batch中的键名
    achieved_goal_key: str = "achieved_goal"  # 达成目标在batch中的键名
    
class BatchAwareHERProcessor:
    """Batch感知的HER处理器"""
    
    def __init__(self, her_config: HERConfig):
        self.config = her_config
        self.batch_stats = {
            'total_batches_processed': 0,
            'successful_batches': 0,
            'her_generated_from_batches': 0,
            'average_batch_reward': 0.0,
            'relabel_operations': 0
        }
    
    def process_batch_for_her(self, batch_rollouts: List[DataProto]) -> List[DataProto]:
        """处理一个batch的rollout用于HER"""
        if len(batch_rollouts) < self.config.min_batch_size_for_her:
            print(f"Batch too small for HER: {len(batch_rollouts)} < {self.config.min_batch_size_for_her}")
            return []
        
        # 计算batch级别的奖励统计
        batch_rewards = self._extract_batch_rewards(batch_rollouts)
        batch_avg_reward = np.mean(batch_rewards)
        batch_success_rate = np.mean([r > self.config.reward_threshold for r in batch_rewards])
        
        self.batch_stats['total_batches_processed'] += 1
        self.batch_stats['average_batch_reward'] = (
            (self.batch_stats['average_batch_reward'] * (self.batch_stats['total_batches_processed'] - 1) + 
             batch_avg_reward) / self.batch_stats['total_batches_processed']
        )
        
        print(f"Batch HER analysis:")
        print(f"  Batch size: {len(batch_rollouts)}")
        print(f"  Average reward: {batch_avg_reward:.3f}")
        print(f"  Success rate: {batch_success_rate:.3f}")
        print(f"  Individual rewards: {batch_rewards}")
        
        # 判断是否需要进行HER处理
        if self._should_apply_her(batch_avg_reward, batch_success_rate):
            her_experiences = self._generate_batch_her_experiences(
                batch_rollouts, batch_avg_reward, batch_success_rate
            )
            self.batch_stats['her_generated_from_batches'] += len(her_experiences)
            
            if batch_avg_reward > self.config.batch_success_threshold:
                self.batch_stats['successful_batches'] += 1
            
            return her_experiences
        
        return []
    
    def _extract_batch_rewards(self, batch_rollouts: List[DataProto]) -> List[float]:
        """提取batch中所有rollout的奖励"""
        rewards = []
        for rollout in batch_rollouts:
            try:
                if 'reward' in rollout.batch:
                    reward = float(rollout.batch['reward'][0].item())
                elif 'acc' in rollout.batch:
                    reward = float(rollout.batch['acc'][0].item())
                else:
                    reward = 0.0
                rewards.append(reward)
            except Exception as e:
                print(f"Failed to extract reward: {e}")
                rewards.append(0.0)
        return rewards
    
    def _should_apply_her(self, batch_avg_reward: float, batch_success_rate: float) -> bool:
        """判断是否应该对这个batch应用HER"""
        if self.config.reward_strategy == HERRewardStrategy.BATCH_AVERAGE:
            # 平均奖励在中等范围时应用HER（既不是完全失败，也不是完全成功）
            return 0.1 < batch_avg_reward < 0.9
        
        elif self.config.reward_strategy == HERRewardStrategy.BATCH_MAJORITY:
            # 成功率在中等范围时应用HER
            return 0.2 < batch_success_rate < 0.8
        
        elif self.config.reward_strategy == HERRewardStrategy.ADAPTIVE:
            # 自适应：根据历史表现调整
            historical_avg = self.batch_stats['average_batch_reward']
            # 如果当前batch表现低于历史平均，应用HER
            return batch_avg_reward < historical_avg
        
        else:  # INDIVIDUAL
            # 传统方式：有失败的rollout就应用HER
            return batch_avg_reward < 1.0
    
    def _generate_batch_her_experiences(self, batch_rollouts: List[DataProto], 
                                       batch_avg_reward: float, 
                                       batch_success_rate: float) -> List[DataProto]:
        """基于batch统计生成HER经验"""
        her_experiences = []
        
        # 找出batch中的成功和失败rollout
        successful_rollouts = []
        failed_rollouts = []
        
        for rollout in batch_rollouts:
            reward = self._get_rollout_reward(rollout)
            if reward > self.config.reward_threshold:
                successful_rollouts.append(rollout)
            else:
                failed_rollouts.append(rollout)
        
        print(f"Batch composition: {len(successful_rollouts)} successful, {len(failed_rollouts)} failed")
        
        # 为失败的rollout生成HER经验
        if successful_rollouts and failed_rollouts:
            her_experiences = self._relabel_failed_rollouts_with_successful_goals(
                failed_rollouts, successful_rollouts, batch_avg_reward
            )
        
        # 如果batch平均奖励较高，为所有rollout生成额外的HER经验
        if batch_avg_reward > self.config.batch_relabel_threshold:
            additional_her = self._generate_high_reward_batch_her(
                batch_rollouts, batch_avg_reward
            )
            her_experiences.extend(additional_her)
        
        return her_experiences
    
    def _get_rollout_reward(self, rollout: DataProto) -> float:
        """获取单个rollout的奖励"""
        try:
            if 'complete' in rollout.batch:
                return float(rollout.batch['complete'][0].item())
            elif 'acc' in rollout.batch:
                return float(rollout.batch['acc'][0].item())
            return 0.0
        except:
            return 0.0
    
    def _relabel_failed_rollouts_with_successful_goals(self, failed_rollouts: List[DataProto], 
                                                      successful_rollouts: List[DataProto],
                                                      batch_avg_reward: float) -> List[DataProto]:
        """使用成功rollout的目标重新标记失败rollout"""
        her_experiences = []
        
        for failed_rollout in failed_rollouts:
            # 为每个失败rollout生成k个HER经验
            for _ in range(self.config.replay_k):
                # 随机选择一个成功rollout作为目标来源
                target_successful = np.random.choice(successful_rollouts)
                
                # 提取成功rollout的目标
                successful_goal = self._extract_achieved_goal_from_rollout(target_successful)
                
                if successful_goal is not None:
                    # 创建HER经验
                    her_rollout = self._create_her_experience_from_batch_context(
                        failed_rollout, successful_goal, batch_avg_reward, target_successful
                    )
                    
                    if her_rollout is not None:
                        her_experiences.append(her_rollout)
        
        print(f"Generated {len(her_experiences)} HER experiences from failed rollouts")
        return her_experiences
    
    def _generate_high_reward_batch_her(self, batch_rollouts: List[DataProto], 
                                       batch_avg_reward: float) -> List[DataProto]:
        """为高奖励batch生成额外的HER经验"""
        her_experiences = []
        
        # 从batch中选择一些rollout进行交叉relabel
        sample_size = min(len(batch_rollouts) // 2, 4)  # 最多选择一半的rollout
        selected_rollouts = np.random.choice(batch_rollouts, sample_size, replace=False)
        
        for rollout in selected_rollouts:
            # 随机选择另一个rollout的目标
            other_rollouts = [r for r in batch_rollouts if r != rollout]
            if other_rollouts:
                target_rollout = np.random.choice(other_rollouts)
                target_goal = self._extract_achieved_goal_from_rollout(target_rollout)
                
                if target_goal is not None:
                    her_rollout = self._create_her_experience_from_batch_context(
                        rollout, target_goal, batch_avg_reward, target_rollout
                    )
                    
                    if her_rollout is not None:
                        her_experiences.append(her_rollout)
        
        print(f"Generated {len(her_experiences)} additional HER experiences from high-reward batch")
        return her_experiences
    
    def _extract_achieved_goal_from_rollout(self, rollout: DataProto) -> Optional[str]:
        """从rollout中提取达成的目标"""
        try:
            # 尝试从输出中提取目标
            if 'output' in rollout.non_tensor_batch:
                output_text = rollout.non_tensor_batch['output'][0]
                return self._extract_goal_from_output(output_text)
            
            # 尝试从其他字段提取
            if self.config.achieved_goal_key in rollout.batch:
                return rollout.batch[self.config.achieved_goal_key][0]
            
            # 使用问题本身作为目标（对于问答任务）
            if 'input' in rollout.non_tensor_batch:
                return rollout.non_tensor_batch['input'][0]
            
            return None
        except Exception as e:
            print(f"Failed to extract goal: {e}")
            return None
    
    def _extract_goal_from_output(self, output_text: str) -> str:
        """从输出文本中提取目标"""
        # 针对0/1奖励任务的目标提取策略
        if not output_text:
            return "generate valid output"
        
        # 对于代码生成任务
        if "def " in output_text or "class " in output_text:
            lines = output_text.strip().split('\n')
            for line in lines:
                if line.strip().startswith(('def ', 'class ')):
                    return f"implement {line.strip()}"
        
        # 对于数学问题
        if any(char in output_text for char in ['=', '+', '-', '*', '/']):
            return "solve mathematical problem correctly"
        
        # 对于一般文本
        sentences = output_text.strip().split('.')
        if sentences:
            return f"generate content like: {sentences[0][:50]}..."
        
        return "generate appropriate response"
    
    def _create_her_experience_from_batch_context(self, original_rollout: DataProto, 
                                                 new_goal: str, 
                                                 batch_avg_reward: float,
                                                 reference_rollout: DataProto) -> Optional[DataProto]:
        """基于batch上下文创建HER经验"""
        try:
            import copy
            her_rollout = copy.deepcopy(original_rollout)
            
            # 重新标记目标
            if self.config.goal_key in her_rollout.non_tensor_batch:
                her_rollout.non_tensor_batch[self.config.goal_key][0] = new_goal
            
            # 重新计算奖励：基于batch上下文和目标匹配度
            new_reward = self._compute_her_reward_with_batch_context(
                her_rollout, new_goal, batch_avg_reward, reference_rollout
            )
            
            # 更新奖励和准确性
            if 'reward' in her_rollout.batch:
                her_rollout.batch['reward'][0] = new_reward
            if 'acc' in her_rollout.batch:
                her_rollout.batch['acc'][0] = new_reward  # 0/1奖励系统
            
            # 标记为HER经验
            her_rollout.meta_info = her_rollout.meta_info or {}
            her_rollout.meta_info.update({
                'is_her': True,
                'her_strategy': self.config.strategy.value,
                'her_reward_strategy': self.config.reward_strategy.value,
                'batch_avg_reward': batch_avg_reward,
                'original_reward': self._get_rollout_reward(original_rollout),
                'relabeled_goal': new_goal
            })
            
            self.batch_stats['relabel_operations'] += 1
            return her_rollout
            
        except Exception as e:
            print(f"HER experience creation error: {e}")
            return None
    
    def _compute_her_reward_with_batch_context(self, rollout: DataProto, 
                                              goal: str, 
                                              batch_avg_reward: float,
                                              reference_rollout: DataProto) -> float:
        """基于batch上下文计算HER奖励"""
        try:
            # 获取rollout的输出
            if 'output' in rollout.non_tensor_batch:
                output = rollout.non_tensor_batch['output'][0]
            else:
                return 0.0
            
            # 获取参考输出
            if 'output' in reference_rollout.non_tensor_batch:
                reference_output = reference_rollout.non_tensor_batch['output'][0]
            else:
                reference_output = ""
            
            # 计算相似度
            similarity = self._compute_output_similarity(output, reference_output, goal)
            
            # 基于batch平均奖励调整阈值
            # 如果batch整体表现好，降低HER奖励阈值；反之提高
            dynamic_threshold = 0.5 + (batch_avg_reward - 0.5) * 0.3
            
            # 0/1奖励：基于相似度和动态阈值
            if similarity > dynamic_threshold:
                return 1.0
            else:
                return 0.0
            
        except Exception as e:
            print(f"HER reward computation error: {e}")
            return 0.0
    
    def _compute_output_similarity(self, output1: str, output2: str, goal: str) -> float:
        """计算输出相似度"""
        try:
            # 简单的词汇重叠相似度
            words1 = set(output1.lower().split())
            words2 = set(output2.lower().split())
            goal_words = set(goal.lower().split())
            
            # 计算与目标的相似度
            goal_similarity = 0.0
            if goal_words:
                overlap_with_goal = words1.intersection(goal_words)
                goal_similarity = len(overlap_with_goal) / len(goal_words)
            
            # 计算与参考输出的相似度
            ref_similarity = 0.0
            if words1 or words2:
                overlap = words1.intersection(words2)
                union = words1.union(words2)
                ref_similarity = len(overlap) / len(union) if union else 0.0
            
            # 综合相似度
            return 0.6 * goal_similarity + 0.4 * ref_similarity
            
        except:
            return 0.0
    
    def get_batch_her_stats(self) -> Dict:
        """获取batch级别的HER统计"""
        return {
            **self.batch_stats,
            'batch_success_rate': (self.batch_stats['successful_batches'] / 
                                  max(1, self.batch_stats['total_batches_processed'])),
            'her_per_batch_ratio': (self.batch_stats['her_generated_from_batches'] / 
                                   max(1, self.batch_stats['total_batches_processed']))
        }

class AsyncExperienceBufferWithHER:
    def __init__(self, config, her_config: Optional[HERConfig] = None):
        self.config = config
        self.her_config = her_config or HERConfig()
        
        # 添加batch级别的HER处理器
        self.batch_her_processor = BatchAwareHERProcessor(self.her_config)
        
        # 批次累积器：收集rollout直到达到batch大小
        self.batch_accumulator = []
        self.target_batch_size = self.her_config.min_batch_size_for_her
        
        self.n_samples = config.data.n_samples
        self.train_batch_size = config.data.train_batch_size
        self.config.data.min_buffer_size = self.n_samples * self.train_batch_size
        
        # 主缓冲区：存储原始经验
        self.original_buffer = deque(maxlen=config.data.max_buffer_size // 2)
        # HER缓冲区：存储重新标记的经验
        self.her_buffer = deque(maxlen=config.data.max_buffer_size // 2)
        self.max_original_size = config.data.max_buffer_size // 2
        self.max_her_size = config.data.max_buffer_size // 2
        
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
        print(f"  Original buffer size: {config.data.max_buffer_size // 2}")
        print(f"  HER buffer size: {config.data.max_buffer_size // 2}")
    
            
    def add_batch(self, batch_data: DataProto):
        """添加批次数据，支持batch级别的HER处理"""
        with self.not_full:
            while self._total_size() >= self.config.data.max_buffer_size:
                self.not_full.wait()
            
            # 将新rollout添加到累积器
            new_rollouts = []
            for i in range(len(batch_data.batch)):
                single_rollout = batch_data[i:i+1]
                
                # 添加到原始缓冲区
                if len(self.original_buffer) >= self.max_original_size:
                    self._remove_oldest_from_buffer(self.original_buffer, self.batch_removal_size)
                self.original_buffer.append(single_rollout)
                
                # 添加到batch累积器用于HER处理
                self.batch_accumulator.append(single_rollout)
                new_rollouts.append(single_rollout)
            
            # 检查是否可以进行batch级别的HER处理
            if len(self.batch_accumulator) >= self.target_batch_size:
                self._process_accumulated_batch_for_her()
            
            print(f"Added {len(batch_data.batch)} rollouts. "
                  f"Accumulator: {len(self.batch_accumulator)}/{self.target_batch_size}")
            print(f"Buffer sizes - Original: {len(self.original_buffer)}, HER: {len(self.her_buffer)}")
            
            self.not_empty.notify_all()
            
    def _remove_oldest_from_buffer(self, buffer, count: int):
        """从指定缓冲区删除最旧的rollout"""
        if len(buffer) <= count:
            buffer.clear()
            return
        
        removed_count = min(count, len(buffer))
        
        for i in range(removed_count):
            buffer.popleft()
        # del buffer[:removed_count] # buffer.slice(start=0, length=removed_count)
        
        print(f"Removed {removed_count} oldest rollouts from buffer")
            
    def _process_accumulated_batch_for_her(self):
        """处理累积的batch进行HER"""
        if not self.her_config.enabled or len(self.batch_accumulator) < self.target_batch_size:
            return
        
        try:
            # 使用batch处理器生成HER经验
            her_experiences = self.batch_her_processor.process_batch_for_her(
                self.batch_accumulator[:self.target_batch_size]
            )
            
            # 将HER经验添加到HER缓冲区
            for her_exp in her_experiences:
                if len(self.her_buffer) >= self.max_her_size:
                    self._remove_oldest_from_buffer(self.her_buffer, self.batch_removal_size)
                
                self.her_buffer.append(her_exp)
            
            # 清理累积器
            self.batch_accumulator = self.batch_accumulator[self.target_batch_size:]
            
            print(f"Processed batch for HER: generated {len(her_experiences)} HER experiences")
            
        except Exception as e:
            print(f"Batch HER processing error: {e}")
            
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
        
        # available_size = min(count, len(buffer))
        # indices = random.sample(range(len(buffer)), available_size)
        # return [buffer[i] for i in indices]
        
        available_size = min(count, len(buffer))
        indices = [-i for i in range(available_size, 0, -1)]
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
        return self._total_size() >= max(min_rollouts_for_training, min_for_batch)
    
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
            if 'complete' in rollout.batch:
                return bool(rollout.batch['complete'][0].item())
            
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
            
            if 'complete' in final_rollout.batch:
                final_reward = final_rollout.batch['complete'][0].item()
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
        
    def get_comprehensive_her_stats(self) -> Dict:
        """获取包含batch级别的综合HER统计"""
        basic_stats = self.get_her_stats()
        batch_stats = self.batch_her_processor.get_batch_her_stats()
        
        return {
            **basic_stats,
            'batch_her_stats': batch_stats,
            'accumulator_size': len(self.batch_accumulator),
            'target_batch_size': self.target_batch_size
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

class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl', action_token_len=7, action_chunks_len=8):
    responses = data.batch['responses']
    
    traj_length = responses.size(1) * action_chunks_len  
    action_length = action_token_len  # next fix
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    #attention_mask = data.batch['attention_mask']
    finish_step = data.batch['finish_step'] * action_length
    
    steps = torch.arange(traj_length*action_length, device=data.batch['responses'].device)  # (traj_len,)
    steps_expanded = steps.unsqueeze(0).expand(data.batch['responses'].size(0), -1)
    response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, gamma, lam, adv_estimator, config):

    responses = data.batch['responses']
    response_length = responses.size(1) *  responses.size(2)
    # attention_mask = data.batch['attention_mask']
    finish_step = data.batch['finish_step'] * config.actor_rollout_ref.model.action_token_len 
    steps = torch.arange(response_length, device=data.batch['responses'].device)  # (traj_len,)
    steps_expanded = steps.unsqueeze(0).expand(data.batch['responses'].size(0), -1)
    response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)

    token_level_rewards = data.batch['token_level_rewards'] if 'token_level_rewards' in list(data.batch.keys()) else data.batch['token_level_scores']

    # TODO: add other ways to estimate advantages
    if adv_estimator == 'rloo':
        # prompt_ids = data.batch['prompts']
        # prompt_length = prompt_ids.shape[-1]
        # valid_response_length = data.batch['attention_mask'][:,prompt_length:].sum(-1)
        advantages, returns = core_algos.compute_rloo_returns(data=data,
                                                eos_mask=response_mask,n_samples=config.data.n_samples, config=config)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(1) *  responses.size(2)
        finish_step = data.batch['finish_step'] * config.actor_rollout_ref.model.action_token_len 
        steps = torch.arange(response_length, device=data.batch['responses'].device)  # (traj_len,)
        steps_expanded = steps.unsqueeze(0).expand(data.batch['responses'].size(0), -1)
        response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'remax':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def compute_data_metrics(batch,config):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    advantages = batch.batch['advantages']
    returns = batch.batch['returns']
    #add
    finish_step = batch.batch['finish_step'] * config.actor_rollout_ref.model.action_token_len 
    steps = torch.arange(batch.batch['responses'].size(1)*batch.batch['responses'].size(2), device=advantages.device)  # (traj_len,)
    steps_expanded = steps.unsqueeze(0).expand(batch.batch['responses'].size(0), -1)
    response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
    #
    metrics = {
        # score
        'critic/score/mean': torch.mean(sequence_score).detach().item(),
        'critic/score/max': torch.max(sequence_score).detach().item(),
        'critic/score/min': torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean': torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max': torch.max(sequence_reward).detach().item(),
        'critic/rewards/min': torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean': masked_mean(advantages, response_mask).detach().item(),
        'critic/advantages/max': torch.max(advantages[response_mask.bool()]).detach().item(),
        'critic/advantages/min': torch.min(advantages[response_mask.bool()]).detach().item(),
        # returns
        'critic/returns/mean': masked_mean(returns, response_mask).detach().item(),
        'critic/returns/max': torch.max(returns[response_mask.bool()]).detach().item(),
        'critic/returns/min': torch.min(returns[response_mask.bool()]).detach().item(),
        # response length
  
    }
    return metrics

class RayTrainer(object):
   
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):


        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping and config.algorithm.kl_ctrl.kl_coef > 0
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        
        self.experience_buffer = AsyncExperienceBuffer(config)
        # self.experience_buffer = AsyncExperienceBufferWithHER(config, HERConfig())
        

    def _create_dataloader(self):   # next fix
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rob_dataset import LIBERO_Dataset, collate_fn
        self.train_dataset = LIBERO_Dataset(self.config.data.task_suite_name,
                                            num_trials_per_task=self.config.data.num_trials_per_task,
                                            train_val ="train")
        self.train_dataloader = BufferedDataLoader(DataLoader(dataset=self.train_dataset,
                                           batch_size=int(8*self.config.data.oversample_factor), #int(self.config.data.train_batch_size*self.config.data.oversample_factor),
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=collate_fn))

        self.val_dataset = LIBERO_Dataset(self.config.data.task_suite_name,
                                        num_trials_per_task=self.config.data.num_trials_per_task,
                                        train_val ="valid")
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self, global_steps=0):
        reward_tensor_lst = []
        data_source_lst = []
        metric_dict = {}
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
           
            test_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
                "global_steps":global_steps
            }

            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_batch)
            print('validation generation end')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            verifier_score, reward_metrics, format_metrics, reward_format_metrics = self.val_reward_fn.verify(test_batch)
            reward_tensor=torch.tensor(verifier_score, dtype=torch.float32).unsqueeze(-1)

            for k, v in reward_metrics.items():
                metric_dict['test_reward/' + k] = v
                
            for k, v in format_metrics.items():
                metric_dict['format_acc/' + k] = v
                
            for k, v in reward_format_metrics.items():
                metric_dict['acc_wformat/' + k] = v
            reward_tensor_lst.append(reward_tensor)
            #data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            data_source_lst.append( [self.config.data.task_suite_name] * reward_tensor.shape[0])

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'test_score/{data_source}'] = np.mean(rewards)

        metric_dict[f'test_score/all'] = reward_tensor.mean().item()

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in ['rloo']:
            self.use_critic = False
        elif self.config.algorithm.adv_estimator in ['grpo']:
            self.use_critic = False
        elif self.config.algorithm.adv_estimator in ['reinforce_plus_plus']:
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def fit(self):
        """
        The training loop of VLA-RL.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        global_steps = 0
        dp_size = self.actor_rollout_wg.world_size // self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        batch_size = self.config.data.train_batch_size
        n_samples = self.config.data.n_samples

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', False):
            val_metrics = self._validate(global_steps=global_steps)
            val_metrics = {f'val/{key}': val for key, val in val_metrics.items()}
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=global_steps)
            if self.config.trainer.get('val_only', False):
                return
            
        # 初始化异步数据生成器
        self.data_generator = AsyncDataGenerator(
            trainer=self,
            actor_rollout_wg=self.actor_rollout_wg,
            tokenizer=self.tokenizer,
            train_dataloader=self.train_dataloader,
            config=self.config,
        )

        for epoch in range(self.config.trainer.total_epochs):
            # self.train_dataloader.start_new_epoch()
            self.data_generator.start_generation()
            while True:
                valid_batch = []
                # buffer_batch = []

                # if self.train_dataloader.buffer_size() > 0:
                #     buffer_batch = self.train_dataloader.get_from_buffer(batch_size, self.actor_rollout_wg.world_size)
                # metrics = defaultdict(list)
                # metrics['timing/gen'] = 0
                # metrics['timing/verify'] = 0
                # metrics['timing/acc&trunc_filter'] = 0
                # metrics['timing/filter_format_error'] = 0
                # metrics['timing/compute_all_entropy'] = 0

                # while len(valid_batch) < batch_size * n_samples:
                #     try:
                #         batch_dict = self.train_dataloader.get_next_batch()
                #     except StopIteration:
                #         break

                #     # generate a batch
                #     with Timer(name='gen', text="{name}: {seconds:.1f} seconds") as timer:

                #         newbatch: DataProto = DataProto.from_single_dict(batch_dict)

                #         if len(buffer_batch) > 0:
                #             newbatch = DataProto.concat([buffer_batch, newbatch])
                #             buffer_batch = []

                #         gen_batch = newbatch.select(batch_keys=['task_id', 'trial_id'],
                #                                     non_tensor_batch_keys={"task_suite_name"},
                #                                     meta_info_keys={})
 
                #         newbatch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(newbatch.batch))],
                #                                              dtype=object)

                #         batch_lst = sum([[newbatch[i:i + 1] for _ in range(n_samples)] for i in range(len(newbatch))],
                #                         [])

                #         gen_batch.meta_info = {
                #             'eos_token_id': self.tokenizer.eos_token_id,
                #             'n_samples': n_samples,
                #             'pad_token_id': self.tokenizer.pad_token_id,
                #         }
                        
                #         gen_batch_output = self.actor_rollout_wg.generate_sequences(prompts=gen_batch)
                        
                #         roll_batch = DataProto.concat(batch_lst)
                #         #roll_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                #         roll_batch = roll_batch.union(gen_batch_output)

                #     metrics['timing/gen'] += timer.last
                    
                    
                    # # do accuracy filtering and score logging
                    # with Timer(name='verify', text="{name}: {seconds:.1f} seconds") as timer:
                    #     scores_tensor, reward_metrics, format_metrics, reward_format_metrics = self.reward_fn.verify(roll_batch)
                    #     for k, v in reward_metrics.items():
                    #         metrics['train_verify_score/' + k].append(v)
                            
                    #     for k, v in format_metrics.items():
                    #         metrics['format_score/' + k].append(v)
                            
                    #     for k, v in reward_format_metrics.items():
                    #         metrics['train_verify_score_wo_format/' + k].append(v)    
                            
                    # metrics['timing/verify'] += timer.last
                    
                    # # do accuracy filtering and score logging
                    # with Timer(name='acc&trunc_filter', text="{name}: {seconds:.1f} seconds") as timer:
                    #     if self.config.data.filter_accuracy or self.config.data.filter_truncated:
                    #         print(f"before filtering: {len(roll_batch)}")
                    #         filtered_roll_batch = self.filter(roll_batch.batch['acc'].unsqueeze(1), roll_batch, n_samples)
                    #         print(f"after filtering: {len(filtered_roll_batch)}")
                    # metrics['timing/acc&trunc_filter'] += timer.last

                    
                    # if self.config.data.filter_warmup:
                    #     raise ValueError
                    #     roll_batch_to_add = filtered_roll_batch if len(filtered_roll_batch) > 0 else roll_batch
                    # else:
                    #     roll_batch_to_add = filtered_roll_batch
                    
                    # if len(valid_batch) == 0:
                    #     valid_batch = roll_batch_to_add
                    # else:
                    #     valid_batch = DataProto.concat([valid_batch, roll_batch_to_add])
                    # print(
                    #     f"collected {len(valid_batch)} / {batch_size * n_samples} rollouts and each prompt has {n_samples} responses")
                
                # 等待缓冲区准备就绪
                # print("Waiting for experience buffer to initialize...")
                while not self.experience_buffer.is_ready():
                    time.sleep(10)
                    buffer_size = self.experience_buffer.size()
                    print(f"Initial buffer size: {buffer_size}/{self.config.data.min_buffer_size}/{self.config.data.max_buffer_size}")
                    
                # 从缓冲区采样批次
                # roll_batch_to_add = self.experience_buffer.sample_batch(batch_size)
                roll_batch_to_add = self.experience_buffer.sample_batch_with_important_sampling(batch_size, sampling_strategy='mean_reward_priority', epsilon=1e-6)
                self.data_generator.traffic_sign = True
                # self.experience_buffer.clear_buffer()
                
                print(f"Collected {len(roll_batch_to_add)} / {batch_size * n_samples} rollouts and each prompt has {n_samples} responses")
                
                if len(valid_batch) == 0:
                    valid_batch = roll_batch_to_add
                else:
                    valid_batch = DataProto.concat([valid_batch, roll_batch_to_add])
                    
                print(f"Current {len(valid_batch)} / {batch_size * n_samples} rollouts and each prompt has {n_samples} responses")
                    
                if len(valid_batch) < batch_size * n_samples:
                    break
                elif len(valid_batch) > batch_size * n_samples:
                    valid_batch = self.add_to_buffer(valid_batch, batch_size, n_samples)
                # import ipdb;ipdb.set_trace()
                # for k, v in reward_metrics.items():
                #     metrics['train_verify_score/' + k] = np.mean(metrics['train_verify_score/' + k])
                    
                # for k, v in format_metrics.items():
                #     metrics['format_score/' + k] = np.mean(metrics['format_score/' + k])
                    
                # for k, v in reward_format_metrics.items():
                #     metrics['train_verify_score_wo_format/' + k] = np.mean(metrics['train_verify_score_wo_format/' + k])
                
                metrics = self.data_generator.get_generation_stats()
                
                batch = valid_batch
                print(f'Start training with rollout batch size: {len(batch)}')
                
                if self.use_reference_policy:
                    # compute reference log_prob
                    with Timer(name='ref', text="{name}: {seconds:.1f} seconds") as timer:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                    metrics['timing/ref'] = timer.last

                with Timer(name='reward', text="{name}: {seconds:.1f} seconds") as timer:
                    if self.use_rm:
                        print("Not implement yet")
                        raise ValueError
                        # batch.meta_info['n_samples'] = n_samples
                        # reward_model_tensor= self.rm_wg.compute_rm_score(batch)
                        # if 'metrics' in reward_model_tensor.meta_info:
                        #     reward_model_metrics = reduce_metrics(reward_model_tensor.meta_info.pop('metrics'))
                        #     metrics.update(reward_model_metrics)
                        # batch = batch.union(reward_model_tensor)

                metrics['timing/reward_model'] = timer.last

                with Timer(name='adv', text="{name}: {seconds:.1f} seconds") as timer:
                    # directly reuse previously computed rewards; but with reward shaping
                    reward_tensor_dict, reward_metrics = self.reward_fn(batch)
                    batch.batch['token_level_scores'] = reward_tensor_dict['all']
                    for k, v in reward_metrics.items():
                        metrics['train_reward/' + k] = v
                    # decomposed rewards:
                    for k,v in reward_tensor_dict.items():
                        batch.batch[k]=v

                    # compute rewards. apply_kl_penalty if available
                    batch, kl_metrics = apply_kl_penalty(batch,
                                                         kl_ctrl=self.kl_ctrl,
                                                         kl_penalty=self.config.algorithm.kl_penalty,
                                                         action_token_len=self.config.actor_rollout_ref.model.action_token_len, 
                                                         action_chunks_len=self.config.actor_rollout_ref.model.action_chunks_len,)
                    metrics.update(kl_metrics)

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(batch,
                                              self.config.algorithm.gamma,
                                              self.config.algorithm.lam,
                                              adv_estimator=self.config.algorithm.adv_estimator,
                                              config = self.config)
                metrics['timing/adv'] = timer.last

                # critic is disabled

                # implement critic warmup
                if self.config.trainer.critic_warmup <= global_steps:
                    # update actor
                    with Timer(name='update_actor', text="{name}: {seconds:.1f} seconds") as timer:
                        batch.meta_info['is_filtered'] = True
                        batch.meta_info['train_mode'] = False
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                        entropy_output = self.actor_rollout_wg.compute_entropy(data=batch)
                    metrics['timing/update_actor'] = timer.last
                    actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                    entropy_output_metrics = reduce_metrics(entropy_output.meta_info['metrics'])
                    metrics.update(actor_output_metrics)
                    metrics.update(entropy_output_metrics)
                self.data_generator.actor_rollout_wg = self.actor_rollout_wg
                self.experience_buffer.set_policy_version(global_steps + 1)
                self.experience_buffer.clear_buffer()
                self.data_generator.traffic_sign = False
                print(f'Finish updating actor')    
                
                # validate
                if self.val_reward_fn is not None and (global_steps + 1) % self.config.trainer.test_freq == 0:
                    with Timer(name='testing', text="{name}: {seconds:.1f} seconds") as timer:
                        val_metrics: dict = self._validate(global_steps=global_steps+1)
                        val_metrics = {f'val/{key}': val for key, val in val_metrics.items()}
                    metrics['timing/testing'] = timer.last
                    metrics.update(val_metrics)
                    logger.log(data=val_metrics, step=global_steps)

                # collect metrics
                with Timer(name='logging1', text="{name}: {seconds:.1f} seconds") as timer:
                    data_metrics = compute_data_metrics(batch=batch, config = self.config)
                with Timer(name='logging2', text="{name}: {seconds:.1f} seconds") as timer:
                    metrics.update(data_metrics)
                with Timer(name='logging3', text="{name}: {seconds:.1f} seconds") as timer:
                    # TODO: make a canonical logger that supports various backend
                    logger.log(data=metrics, step=global_steps)

                if self.config.trainer.save_freq > 0 and (global_steps + 1) % self.config.trainer.save_freq == 0:
                    actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                                    f'global_step_{global_steps}')
                    actor_remote_path = None #if self.config.trainer.default_hdfs_dir is None else os.path.join(
                        # self.config.trainer.default_hdfs_dir, 'actor')
                    self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

                    if self.use_critic:
                        critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                                         f'global_step_{global_steps}')
                        critic_remote_path = None #if self.config.trainer.default_hdfs_dir is None else os.path.join(
                            # self.config.trainer.default_hdfs_dir, 'critic')
                        self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)
                    if self.use_rm:
                        prm_local_path = os.path.join(self.config.trainer.default_local_dir, 'prm',
                                                         f'global_step_{global_steps}')
                        prm_remote_path = None #if self.config.trainer.default_hdfs_dir is None else os.path.join(
                            # self.config.trainer.default_hdfs_dir, 'critic')
                        self.rm_wg.save_checkpoint(prm_local_path, prm_remote_path)

                global_steps += 1

        # perform validation after training
        if self.val_reward_fn is not None:
            val_metrics = self._validate(global_steps=global_steps)
            pprint(f'Final validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=global_steps)

    def filter_format(self, reward_tensor, batch, n_samples):
        """
        Filter responses based on accuracy and truncation criteria.
        
        Args:
            reward_tensor: Tensor containing accuracy scores
            batch: DataProto batch containing responses
            n_samples: Number of responses per prompt
        
        Returns:
            DataProto: Filtered batch
        """
        if self.config.data.filter_format:
            reward_matrix = reward_tensor.sum(-1).reshape(-1, n_samples)
            acc_tensor = torch.mean(reward_matrix, dim=-1)
            counts = Counter(acc_tensor.tolist())
            print("Format distribution:", " ".join(f"{k:.2f}:{v}" for k, v in sorted(counts.items())))

            acc_mask = (acc_tensor >= 1)
        else:
            # If accuracy filtering disabled, keep all samples
            acc_mask = torch.ones(len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device)
        # Then do truncation filtering if enabled

        # Combine both masks
        combined_mask = acc_mask

        # Expand mask to cover all samples for each prompt
        final_mask = combined_mask.repeat_interleave(n_samples)

        # Apply the mask to the batch
        filtered_batch = batch.slice(final_mask)

        print(f"Filtered format batch size: {len(filtered_batch)} (from original size: {len(batch)})")
        
        return filtered_batch

    def filter(self, reward_tensor, batch, n_samples):
        """
        Filter responses based on accuracy and truncation criteria.
        
        Args:
            reward_tensor: Tensor containing accuracy scores
            batch: DataProto batch containing responses
            n_samples: Number of responses per prompt
        
        Returns:
            DataProto: Filtered batch
        """
        # First do accuracy filtering if enabled
        if self.config.data.filter_accuracy:
            reward_matrix = reward_tensor.sum(-1).reshape(-1, n_samples)
            acc_tensor = torch.mean(reward_matrix, dim=-1)
            counts = Counter(acc_tensor.tolist())
            print("Accuracy distribution:", " ".join(f"{k:.2f}:{v}" for k, v in sorted(counts.items())))

            acc_mask = (acc_tensor >= self.config.data.accuracy_lower_bound) & (
                        acc_tensor <= self.config.data.accuracy_upper_bound)
        else:
            # If accuracy filtering disabled, keep all samples
            acc_mask = torch.ones(len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device)
        # Then do truncation filtering if enabled
        if self.config.data.filter_truncated:
            responses = batch.batch['responses']
            attention_mask = batch.batch['attention_mask']
            response_mask = attention_mask[:, -responses.size(1):]

            # Calculate response lengths
            response_lengths = response_mask.sum(-1)  # (batch_size,)
            response_lengths = response_lengths.reshape(-1, n_samples)  # (num_prompts, n_samples)

            # Get max possible length from config
            max_len = self.config.data.max_response_length

            # Check if any response in the group hits max length (indicating possible truncation)
            has_truncated = (response_lengths >= max_len).any(dim=-1)

            # Print distribution of truncated vs non-truncated
            truncated_counts = Counter(has_truncated.tolist())
            print("Truncation distribution:", 
                f"Truncated: {truncated_counts[True] if True in truncated_counts else 0}, "
                f"Non-truncated: {truncated_counts[False] if False in truncated_counts else 0}")
            # Keep only prompts where no response was truncated
            trunc_mask = ~has_truncated
        else:
            # If truncation filtering disabled, keep all samples
            trunc_mask = torch.ones(len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device)

        # Combine both masks
        combined_mask = acc_mask & trunc_mask

        # Expand mask to cover all samples for each prompt
        final_mask = combined_mask.repeat_interleave(n_samples)

        # Apply the mask to the batch
        filtered_batch = batch.slice(final_mask)

        print(f"Filtered batch size: {len(filtered_batch)} (from original size: {len(batch)})")
        return filtered_batch

    def add_to_buffer(self, batch, batch_size, n_samples):
        buffer_length = len(batch) // n_samples - batch_size
        # buffer_batch = batch.slice(range(batch_size * n_samples, (buffer_length + batch_size) * n_samples, n_samples))
        # # notice that we only add prompts to buffer, and slicing strategy should be exactly consistent to what is in ray_trainer.py
        # buffer_batch = buffer_batch.select(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
        # buffer_batch.slice_batch(start=0, length=self.config.data.max_prompt_length, dim=1)
        buffer_mask = torch.ones(buffer_length + batch_size, dtype=torch.bool)
        buffer_mask[batch_size:] = False
        buffer_mask = buffer_mask.repeat_interleave(n_samples)
        batch = batch.slice(buffer_mask)
        # self.train_dataloader.add_to_buffer(buffer_batch)
        return batch
