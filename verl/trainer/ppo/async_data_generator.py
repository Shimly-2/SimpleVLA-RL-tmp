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
        self.batch_size = config.data.train_batch_size  # 32
        
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
        """异步生成主循环"""
        while not self.stop_event.is_set():
            # try:
            # 检查experience buffer是否需要更多数据
            if (hasattr(self.trainer, 'experience_buffer') and 
                self.trainer.experience_buffer.size() >= self.trainer.config.data.max_buffer_size * 0.8):
                await asyncio.sleep(0.1)
                continue
            
            # 异步生成批次
            generated_batch = await self._generate_batch_async()
            print("generated_batch", generated_batch)
            
            if generated_batch is not None:
                # 添加到experience buffer
                if hasattr(self.trainer, 'experience_buffer'):
                    self.trainer.experience_buffer.add_batch(generated_batch)
                
                # 更新统计
                self.total_generated += len(generated_batch.batch)
                
                # 可选：限制生成速率
                await asyncio.sleep(0.01)
                    
            # except Exception as e:
            #     print(f"Generation loop error: {e}")
            #     await asyncio.sleep(1)
                
    
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
        start_time = time.time()
        
        try:
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
            gen_time = time.time() - start_time
            batch_metrics = {
                'timing/gen': gen_time,
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
        """异步生成单个批次数据 - 包含完整的验证和过滤流程"""
        try:
            # 收集经过验证和过滤的rollout数据
            valid_batch = []
            buffer_batch = []
            
            # 从dataloader buffer获取数据
            if self.train_dataloader.buffer_size() > 0:
                buffer_batch = await self._get_from_buffer_async(
                    self.batch_size, self.actor_rollout_wg.world_size
                )
            
            # 准备指标收集
            all_metrics = defaultdict(list)
            
            # 目标：收集足够的rollout（考虑过滤后的数量）
            target_rollouts = self.batch_size * self.n_samples
            collected_rollouts = 0
            generation_attempts = 0
            max_attempts = target_rollouts * 2  # 考虑过滤率，允许更多尝试
            
            while collected_rollouts < target_rollouts and generation_attempts < max_attempts:
                # 获取下一个原始批次数据
                batch_dict = await self._get_next_batch_async()
                if batch_dict is None:
                    print("No more data available from dataloader")
                    break
                
                generation_attempts += 1
                
                # 处理这个批次（包含生成、验证、过滤）
                processed_batch, batch_metrics = await self._process_single_batch_with_nsamples(
                    batch_dict, buffer_batch
                )
                
                if processed_batch is not None and len(processed_batch.batch) > 0:
                    valid_batch.append(processed_batch)
                    collected_rollouts += len(processed_batch.batch)
                    
                    # 收集指标
                    for key, value in batch_metrics.items():
                        if isinstance(value, (list, tuple)):
                            all_metrics[key].extend(value)
                        else:
                            all_metrics[key].append(value)
                    
                    # 清空buffer_batch（只在第一次使用）
                    if len(buffer_batch) > 0:
                        buffer_batch = []
                    
                    print(f"Collected {collected_rollouts}/{target_rollouts} rollouts "
                          f"(attempts: {generation_attempts})")
                else:
                    print(f"Batch processing failed or resulted in empty batch (attempt {generation_attempts})")
            
            if valid_batch:
                # 合并所有有效批次
                final_batch = DataProto.concat(valid_batch)
                
                # 如果超过了目标数量，截取到正确的大小
                if len(final_batch.batch) > target_rollouts:
                    final_batch = final_batch[:target_rollouts]
                
                # 计算总体统计
                total_filter_pass_rate = np.mean(all_metrics.get('filter_pass_rate', [0]))
                total_generation_time = sum(all_metrics.get('timing/gen', []))
                total_verify_time = sum(all_metrics.get('timing/verify', []))
                total_filter_time = sum(all_metrics.get('timing/acc&trunc_filter', []))
                
                print(f"Generation batch completed:")
                print(f"  Final rollouts: {len(final_batch.batch)}")
                print(f"  Generation attempts: {generation_attempts}")
                print(f"  Average filter pass rate: {total_filter_pass_rate:.2%}")
                print(f"  Total times - Gen: {total_generation_time:.1f}s, "
                      f"Verify: {total_verify_time:.1f}s, Filter: {total_filter_time:.1f}s")
                
                # 更新全局指标
                for key, values in all_metrics.items():
                    self.generation_metrics[key].extend(values if isinstance(values, list) else [values])
                
                # 更新统计
                self.total_generated += len(final_batch.batch)
                
                return final_batch
            
            print(f"Failed to generate sufficient rollouts: {collected_rollouts}/{target_rollouts}")
            return None
            
        except Exception as e:
            print(f"Generate batch with filtering error: {e}")
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
            combined_metrics['total_processing_time'] = time.time() - start_time
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
        start_time = time.time()
        
        try:
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
            gen_time = time.time() - start_time
            gen_metrics = {
                'timing/gen': gen_time,
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