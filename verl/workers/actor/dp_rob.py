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

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, log_probs_from_logits_all_rmpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F
from codetiming import Timer
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['RobDataParallelPPOActor']



class RobDataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        print(f'PRM use dynamic bsz={self.config.get("use_dynamic_bsz", False)}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = False #self.ulysses_sequence_parallel_size > 1
        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
       
    def process_tensor(self, tensor, pad_id):
        mask = tensor != pad_id
        if not torch.all(mask == mask[0:1], dim=1).all():
            raise ValueError("Padding error!")
        base_mask = mask[0]
        valid_len = base_mask.sum().item()
        return tensor[:, base_mask], valid_len
    
    def generate_traj_mask(self, end_step, traj_len):
        """
        Args:
            end_step: (batch_size,), 
            traj_len: 
        Returns:
            mask: (batch_size, traj_len),
        """
        steps = torch.arange(traj_len, device=end_step.device)  # (traj_len,)
        steps_expanded = steps.unsqueeze(0).expand(end_step.size(0), -1)
        mask = steps_expanded < end_step.unsqueeze(1)  # (batch_size, traj_len)
        return mask
    
    def apply_mask_with_grad_control(self, log_probs, entropy, mask):
        """
        Args:
            log_probs: (batch_size, traj_len, ...)
            entropy:   (batch_size, traj_len, ...)
            mask:      (batch_size, traj_len)
        Returns:
            log_probs_masked: 
            entropy_masked:   
        """
        mask_expanded = mask.unsqueeze(-1)  

        log_probs_masked = torch.where(
            mask_expanded,
            log_probs,
            torch.zeros_like(log_probs, requires_grad=False)  
        )

        entropy_masked = torch.where(
            mask_expanded,
            entropy,
            torch.zeros_like(entropy, requires_grad=False)   
        )

        return log_probs_masked, entropy_masked

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        micro_batch:
        
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        
        batch_size = micro_batch['responses'].size(0)
        traj_len = micro_batch['responses'].size(1)
        tot_pad_len = micro_batch['input_ids'].size(2)
        
        assert all(micro_batch[key].size(0) == batch_size for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(1) == traj_len for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(2) == tot_pad_len for key in [ 'input_ids', 'attention_mask'])
        
            
        response_length = micro_batch['responses'].size(-1) # 7*8
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            attention_mask = micro_batch['attention_mask']
            pixel_values = micro_batch["pixel_values"]
            responses = micro_batch["responses"]
            
            input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
            attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
            pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
            responses = responses.reshape((batch_size * traj_len,) + responses.shape[2:])
            
            input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
            attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)
            
            if self.config.vla == "openvla-oft":
                logits = self.actor_module(input_ids=input_ids_unpad,
                                        attention_mask=attention_mask_unpad,
                                        pixel_values=pixel_values,
                                        )  # prevent model thinks we are generating
                
                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256 
                logits = logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
                responses = responses - start_index
                #assert (0<=responses<=255).all()
            
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
            
                assert len(log_probs.shape)==2 and len(entropy.shape)==2 
                log_probs = log_probs.reshape((batch_size, traj_len*8,7) )
                entropy = entropy.reshape((batch_size, traj_len*8,7) )

                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len*8)
                log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)
                
                log_probs = log_probs.reshape((batch_size, traj_len*response_length))
                entropy = entropy.reshape((batch_size, traj_len*response_length)) 
                
            elif self.config.vla == "openvla":
                output = self.actor_module(input_ids=input_ids_unpad,
                                    attention_mask=attention_mask_unpad,
                                    pixel_values=pixel_values,
                                    use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                #ADD
                
                log_probs = log_probs.reshape((batch_size, traj_len,) + log_probs.shape[1:])
                entropy = entropy.reshape((batch_size, traj_len,) + entropy.shape[1:])

                
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len)
                log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)
                
                log_probs = log_probs.reshape((batch_size, traj_len*response_length))
                entropy = entropy.reshape((batch_size, traj_len*response_length))
                
                

            return entropy, log_probs
    
    def _forward_micro_batch_update(self, input_ids, attention_mask, pixel_values, responses, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
       
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if self.config.vla == "openvla-oft":
                
                input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)

                
                logits = self.actor_module(input_ids=input_ids_unpad,
                                                attention_mask=attention_mask_unpad,
                                                pixel_values=pixel_values,
                                                )  
                
                assert logits.requires_grad 
                
                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256 
                logits = logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
                responses = responses - start_index
                
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                
                log_probs = log_probs.reshape((1, -1))
                entropy = entropy.reshape((1, -1))
                
                return entropy, log_probs
            
            elif self.config.vla == "openvla":
                response_length = responses.size(-1)
                input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)
                output = self.actor_module(input_ids=input_ids_unpad,
                                        attention_mask=attention_mask_unpad,
                                        pixel_values=pixel_values,
                                        use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                #
                
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                
                
                log_probs = log_probs.reshape((1, -1))
                entropy = entropy.reshape((1, -1))

                return entropy, log_probs
                

    def _forward_micro_batch_entropy(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = micro_batch['responses'].size(0)
        traj_len = micro_batch['responses'].size(1)
        tot_pad_len = micro_batch['input_ids'].size(2)
 
        assert all(micro_batch[key].size(0) == batch_size for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(1) == traj_len for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(2) == tot_pad_len for key in [ 'input_ids', 'attention_mask'])
            
        response_length = micro_batch['responses'].size(-1)
        #assert response_length == 7*8
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            #batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            pixel_values = micro_batch["pixel_values"]
            
            input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
            attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
            pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
            
            
            input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
            attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)

            if  self.config.vla == "openvla-oft":
            
                logits = self.actor_module(input_ids=input_ids_unpad,
                                                attention_mask=attention_mask_unpad,
                                                pixel_values=pixel_values,
                                                ) 
            
                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256 
                logits = logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
            
                logits = logits.div(temperature) 
            
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                assert len(entropy.shape)==2 
                entropy = entropy.reshape((batch_size, traj_len*8,7) )
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len*8)
                _, entropy = self.apply_mask_with_grad_control(entropy, entropy, mask)
                entropy = entropy.reshape((batch_size, traj_len*response_length))
                return entropy
            
            elif self.config.vla == "openvla":
                output = self.actor_module(input_ids=input_ids_unpad,
                                        attention_mask=attention_mask_unpad,
                                        pixel_values=pixel_values,
                                        use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                #
                
                
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div(temperature) 
                
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                #ADD

                entropy = entropy.reshape((batch_size, traj_len,) + entropy.shape[1:])
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len)
                _, entropy = self.apply_mask_with_grad_control(entropy, entropy, mask)
                entropy = entropy.reshape((batch_size, traj_len*response_length))
                return entropy


    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size'] #256
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error # 1
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz'] #trues
        self.pad_token_id = data.meta_info['pad_token_id']
        
        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values',"finish_step"]
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs
    
    # modified: get the top top_ratio of entropy mask
    def get_top_entropy_mask_simple(self, entropy, response_mask, top_ratio=0.2):
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

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values', 'old_log_probs', 'advantages', 'finish_step', 'off_policy_mask', 'on_policy_mask']
        batch = data.select(batch_keys=select_keys).batch
        assert self.config.ppo_micro_batch_size == 1

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for test_idx, data in enumerate(micro_batches):
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                
                response_length = responses.size(1) *  responses.size(2)
                finish_step = data['finish_step'] * self.config.action_token_len
                steps = torch.arange(response_length, device=data['responses'].device)  # (traj_len,)
                steps_expanded = steps.unsqueeze(0).expand(data['responses'].size(0), -1)
                response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
                
                response_mask_sum = response_mask.sum(axis=None)

                old_log_prob = data['old_log_probs']
                advantages = data['advantages']
                
                #clip_ratio = self.config.clip_ratio
                clip_ratio_high = self.config.clip_ratio_high
                clip_ratio_low = self.config.clip_ratio_low
                entropy_coeff = self.config.entropy_coeff
                loss_agg_mode = self.config.loss_agg_mode

                batch_size = data['responses'].size(0)
                traj_len = data['responses'].size(1)
                tot_pad_len = data['input_ids'].size(2)
                
                
                input_ids = data['input_ids']
                attention_mask = data['attention_mask']
                pixel_values = data["pixel_values"]
                responses = data["responses"]
                off_policy_mask = data['off_policy_mask']
                on_policy_mask = data['on_policy_mask']
                          
                input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
                attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
                pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
                responses = responses.reshape((batch_size * traj_len,) + responses.shape[2:])
          
                loss_info = {
                    #'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss':0,
                    'actor/pg_clipfrac': 0,
                    'actor/ppo_kl': 0,
                    'actor/entropy': 0,
                    # modified: add metrics with different mask
                    "actor/advantages_high_entropy": 0,
                    "actor/advantages_low_entropy": 0,
                    "actor/entropy_high": 0,
                    "actor/entropy_low": 0,
                    "actor/cov_high_entropy": 0,
                    "actor/cov_low_entropy": 0,
                    "actor/ratio_high_entropy": 0,
                    "actor/ratio_low_entropy": 0,
                    
                    "actor/off_pg_loss": 0,
                    "actor/on_pg_loss": 0,
                    "actor/off_pg_clipfrac": 0,
                    "actor/off_policy_prob": 0,
                    "actor/on_policy_prob": 0,
                    "actor/off_ratio_mean": 0,
                    "actor/off_ratio_max_clip_frac": 0,
                    "actor/off_ratio_min_clip_frac": 0,
                    "actor/on_ratio_mean": 0,
                    "actor/on_ratio_max_clip_frac": 0,
                    "actor/on_ratio_min_clip_frac": 0,
                }
                
                assert traj_len % self.config.traj_mini_batch_size ==0
                traj_split_num = int(traj_len/self.config.traj_mini_batch_size)
                
                
    

                for i in range(0, traj_len, int(traj_len/traj_split_num)):
                   
                    entropy, log_prob = self._forward_micro_batch_update(input_ids=input_ids[i:i+int(traj_len/traj_split_num)], attention_mask=attention_mask[i:i+int(traj_len/traj_split_num)], pixel_values=pixel_values[i:i+int(traj_len/traj_split_num)], responses=responses[i:i+int(traj_len/traj_split_num)], temperature=temperature)
                    
                    slice_id = i*self.config.action_token_len*self.config.action_chunks_len
                    next_slice_id = (i+int(traj_len/traj_split_num))*self.config.action_token_len*self.config.action_chunks_len
                    old_log_prob_tmp = old_log_prob[:, slice_id: next_slice_id]
                    advantages_tmp = advantages[:, slice_id: next_slice_id]
                    response_mask_tmp = response_mask[:, slice_id: next_slice_id]
                    # prefix_mask_tmp = 
                    
                    # data_dict = {
                    #     'i': i,
                    #     'i+1': i+int(traj_len/traj_split_num),
                    #     'traj_len': traj_len,
                    #     'traj_split_num': traj_split_num,
                    #     'traj_mini_batch_size': self.config.traj_mini_batch_size,
                    #     'test_idx': test_idx,
                    #     'slice_id': slice_id,
                    #     'entropy': entropy,
                    #     'log_prob': log_prob,
                    #     'old_log_prob_tmp': old_log_prob_tmp,
                    #     'advantages_tmp': advantages_tmp,
                    #     'response_mask_tmp': response_mask_tmp,
                    #     # 'importance_weights_tmp': importance_weights_tmp,
                    #     'input_ids': input_ids,
                    #     'input_ids_train': input_ids[i:i+int(traj_len/traj_split_num)],
                    #     'response_mask_sum': response_mask_sum,
                    #     'response_mask': response_mask,
                    #     'on_policy_mask': on_policy_mask,
                    #     'off_policy_mask': off_policy_mask,
                    # }
                    # torch.save(data_dict, f'/share/home/u16023/jinyiyang/experiments/SimpleVLA-RL/outputs/tensors_b{batch_idx}_t{test_idx}_s{slice_id}_wmask.pt')
                    
                    # modified: find top_ratio of entropy and less
                    # Cov
                    cov = (advantages_tmp - advantages_tmp[response_mask_tmp].mean()) * (log_prob - log_prob[response_mask_tmp].mean()) * response_mask_tmp
                    # ratio
                    negative_approx_kl = log_prob - old_log_prob_tmp
                    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
                    ratio = torch.exp(negative_approx_kl)
                    if entropy_coeff != 0:
                        response_mask_tmp_more_than_threshold, mask_entropy_less_than_threshold = self.get_top_entropy_mask_simple(entropy, response_mask_tmp, top_ratio=0.2)
                        # lof metrics with different mask
                        advantages_high_entropy = agg_loss(loss_mat=advantages_tmp, loss_mask=response_mask_tmp_more_than_threshold, loss_agg_mode=loss_agg_mode)
                        advantages_low_entropy = agg_loss(loss_mat=advantages_tmp, loss_mask=mask_entropy_less_than_threshold, loss_agg_mode=loss_agg_mode)
                        entropy_high = agg_loss(loss_mat=entropy, loss_mask=response_mask_tmp_more_than_threshold, loss_agg_mode=loss_agg_mode)
                        entropy_low = agg_loss(loss_mat=entropy, loss_mask=mask_entropy_less_than_threshold, loss_agg_mode=loss_agg_mode)
                        cov_high_entropy = agg_loss(loss_mat=cov, loss_mask=response_mask_tmp_more_than_threshold, loss_agg_mode=loss_agg_mode)
                        cov_low_entropy = agg_loss(loss_mat=cov, loss_mask=mask_entropy_less_than_threshold, loss_agg_mode=loss_agg_mode)
                        ratio_high_entropy = agg_loss(loss_mat=ratio, loss_mask=response_mask_tmp_more_than_threshold, loss_agg_mode=loss_agg_mode)
                        ratio_low_entropy = agg_loss(loss_mat=ratio, loss_mask=mask_entropy_less_than_threshold, loss_agg_mode=loss_agg_mode)
                    else:
                        advantages_high_entropy = torch.tensor(0.0, device=advantages_tmp.device)
                        advantages_low_entropy = torch.tensor(0.0, device=advantages_tmp.device)
                        entropy_high = torch.tensor(0.0, device=advantages_tmp.device)
                        entropy_low = torch.tensor(0.0, device=advantages_tmp.device)
                        cov_high_entropy = torch.tensor(0.0, device=advantages_tmp.device)
                        cov_low_entropy = torch.tensor(0.0, device=advantages_tmp.device)
                        ratio_high_entropy = torch.tensor(0.0, device=advantages_tmp.device)
                        ratio_low_entropy = torch.tensor(0.0, device=advantages_tmp.device)
                        
                    # pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob_tmp,
                    #                                                         log_prob=log_prob,
                    #                                                         advantages=advantages_tmp,
                    #                                                         eos_mask=response_mask_tmp,
                    #                                                         clip_ratio_high=clip_ratio_high,
                    #                                                         clip_ratio_low=clip_ratio_low)
                    
                    if True: #self.config.use_off_policy_loss:
                        # from .mix_core_alg import compute_token_on_off_policy_loss
                        # loss_fn = compute_token_on_off_policy_loss
                        # print(off_policy_mask, off_policy_mask.item())
                        ret_dict = core_algos.compute_token_on_off_policy_loss_v2(old_log_prob=old_log_prob_tmp, 
                            log_prob=log_prob,
                            advantages=advantages_tmp,
                            eos_mask=response_mask_tmp,
                            cliprange=clip_ratio_low,
                            clip_upper_bound=clip_ratio_high,
                            prefix_mask=off_policy_mask,
                            off_cliprange=self.config.off_policy_cliprange,
                            off_normalize=self.config.off_policy_normalize,
                            off_max_clip=-self.config.off_policy_max_clip if self.config.off_policy_max_clip != -1 else None,
                            off_min_clip=-self.config.off_policy_min_clip if self.config.off_policy_min_clip != -1 else None,
                            all_max_clip=-self.config.all_max_clip if self.config.all_max_clip != -1 else None,
                            off_policy_reshape=self.config.off_policy_reshape,
                            off_policy_reshape_weight=self.config.off_policy_reshape_weight,
                            off_policy_reshape_pow_exp=self.config.off_policy_reshape_pow_exp,
                            on_policy_reshape=self.config.on_policy_reshape,
                            on_policy_reshape_weight=self.config.on_policy_reshape_weight,
                            on_policy_reshape_pow_exp=self.config.on_policy_reshape_pow_exp,
                            target_probs=data['target_probs'] if 'target_probs' in data else None,
                            loss_remove_token_mean=self.config.loss_remove_token_mean,
                            loss_remove_clip=self.config.loss_remove_clip
                        )
                        pg_loss = ret_dict['pg_loss']
                        off_pg_loss = ret_dict['off_pg_loss']
                        on_pg_loss = ret_dict['on_pg_loss']
                        off_pg_clipfrac = ret_dict['off_pg_clipfrac']
                        pg_clipfrac = ret_dict['on_pg_clipfrac']
                        ppo_kl = ret_dict['ppo_kl']
                        # print(ret_dict['on_ratio_mean'], pg_loss, off_pg_loss, on_pg_loss)
                        
                        # data = {
                        #     'actor/off_pg_loss': off_pg_loss.detach().item(),
                        #     'actor/on_pg_loss': on_pg_loss.detach().item(),
                        #     'actor/off_pg_clipfrac': off_pg_clipfrac.detach().item(),
                        # }
                        # if 'off_policy_prob' in ret_dict:
                        #     data['actor/off_policy_prob'] = ret_dict['off_policy_prob'].detach().item()
                        # if 'on_policy_prob' in ret_dict:
                        #     data['actor/on_policy_prob'] = ret_dict['on_policy_prob'].detach().item()
                        # if 'off_ratio_mean' in ret_dict:
                        #     data['actor/off_ratio_mean'] = ret_dict['off_ratio_mean'].detach().item()
                        # if 'off_ratio_max_clip_frac' in ret_dict:
                        #     data['actor/off_ratio_max_clip_frac'] = ret_dict['off_ratio_max_clip_frac'].detach().item()
                        # if 'off_ratio_min_clip_frac' in ret_dict:
                        #     data['actor/off_ratio_min_clip_frac'] = ret_dict['off_ratio_min_clip_frac'].detach().item()
                        # append_to_dict(metrics, data)
                    
                    response_mask_tmp_sum = response_mask_tmp.sum(axis=None)
                    pg_loss = pg_loss* response_mask_tmp_sum
                    pg_clipfrac = pg_clipfrac* response_mask_tmp_sum / response_mask_sum
                    ppo_kl = ppo_kl* response_mask_tmp_sum / response_mask_sum
                    
                    # policy_loss = pg_loss / response_mask_sum
                    
                    # compute entropy loss from entropy
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask_tmp, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss / response_mask_sum - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss / response_mask_sum
                        # entropy_loss = torch.zeros_like(policy_loss)
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask_tmp, loss_agg_mode=loss_agg_mode)
                    
                    loss = policy_loss / self.gradient_accumulation
                    
                    loss.backward()
                    
                    loss_info['actor/pg_loss'] =  loss_info['actor/pg_loss'] + policy_loss.detach().item()
                    loss_info['actor/pg_clipfrac'] = loss_info['actor/pg_clipfrac'] + pg_clipfrac.detach().item()
                    loss_info['actor/ppo_kl'] = loss_info['actor/ppo_kl'] +  ppo_kl.detach().item()
                    loss_info['actor/entropy'] = loss_info['actor/entropy'] +  entropy_loss.detach().item()
                    # modified: add metrics with different mask
                    loss_info['actor/advantages_high_entropy'] = loss_info['actor/advantages_high_entropy'] +  advantages_high_entropy.detach().item()
                    loss_info['actor/advantages_low_entropy'] = loss_info['actor/advantages_low_entropy'] +  advantages_low_entropy.detach().item()
                    loss_info['actor/entropy_high'] = loss_info['actor/entropy_high'] +  entropy_high.detach().item()
                    loss_info['actor/entropy_low'] = loss_info['actor/entropy_low'] +  entropy_low.detach().item()
                    loss_info['actor/cov_high_entropy'] = loss_info['actor/cov_high_entropy'] +  cov_high_entropy.detach().item()
                    loss_info['actor/cov_low_entropy'] = loss_info['actor/cov_low_entropy'] +  cov_low_entropy.detach().item()
                    loss_info['actor/ratio_high_entropy'] = loss_info['actor/ratio_high_entropy'] +  ratio_high_entropy.detach().item()
                    loss_info['actor/ratio_low_entropy'] = loss_info['actor/ratio_low_entropy'] +  ratio_low_entropy.detach().item()
                    
                    loss_info['actor/off_pg_loss'] = loss_info['actor/off_pg_loss'] +  off_pg_loss.detach().item()
                    loss_info['actor/on_pg_loss'] = loss_info['actor/on_pg_loss'] +  on_pg_loss.detach().item()
                    if 'off_policy_prob' in ret_dict:
                        loss_info['actor/off_policy_prob'] = loss_info['actor/off_policy_prob'] +  ret_dict['off_policy_prob'].detach().item()
                    if 'on_policy_prob' in ret_dict:
                        loss_info['actor/on_policy_prob'] = loss_info['actor/on_policy_prob'] +  ret_dict['on_policy_prob'].detach().item()
                    if 'off_ratio_mean' in ret_dict:
                        loss_info['actor/off_ratio_mean'] = loss_info['actor/off_ratio_mean'] +  ret_dict['off_ratio_mean'].detach().item()
                    if 'off_ratio_max_clip_frac' in ret_dict:
                        loss_info['actor/off_ratio_max_clip_frac'] = loss_info['actor/off_ratio_max_clip_frac'] +  ret_dict['off_ratio_max_clip_frac'].detach().item()
                    if 'off_ratio_min_clip_frac' in ret_dict:
                        loss_info['actor/off_ratio_min_clip_frac'] = loss_info['actor/off_ratio_min_clip_frac'] +  ret_dict['off_ratio_min_clip_frac'].detach().item()
                    if 'on_ratio_mean' in ret_dict:
                        loss_info['actor/on_ratio_mean'] = loss_info['actor/on_ratio_mean'] +  ret_dict['on_ratio_mean'].detach().item()
                    if 'on_ratio_max_clip_frac' in ret_dict:
                        loss_info['actor/on_ratio_max_clip_frac'] = loss_info['actor/on_ratio_max_clip_frac'] +  ret_dict['on_ratio_max_clip_frac'].detach().item()
                    if 'on_ratio_min_clip_frac' in ret_dict:
                        loss_info['actor/on_ratio_min_clip_frac'] = loss_info['actor/on_ratio_min_clip_frac'] +  ret_dict['on_ratio_min_clip_frac'].detach().item()
                
                append_to_dict(metrics, loss_info)
                
                # entropy, log_prob = [], []
                # for i in range(0, traj_len, int(traj_len/traj_split_num)):
                   
                #     entropy_tmp, log_prob_tmp = self._forward_micro_batch_update(input_ids=input_ids[i:i+int(traj_len/traj_split_num)], attention_mask=attention_mask[i:i+int(traj_len/traj_split_num)], pixel_values=pixel_values[i:i+int(traj_len/traj_split_num)], responses=responses[i:i+int(traj_len/traj_split_num)], temperature=temperature)
                #     entropy.append(entropy_tmp)
                #     log_prob.append(log_prob_tmp)
                #     # slice_id = i*self.config.action_token_len*self.config.action_chunks_len
                #     # next_slice_id = (i+int(traj_len/traj_split_num))*self.config.action_token_len*self.config.action_chunks_len
                #     # old_log_prob_tmp = old_log_prob[:, slice_id: next_slice_id]
                #     # advantages_tmp = advantages[:, slice_id: next_slice_id]
                #     # response_mask_tmp = response_mask[:, slice_id: next_slice_id]
                # entropy = torch.cat(entropy, dim=1)
                # log_prob = torch.cat(log_prob, dim=1)
                # # modified: find top_ratio of entropy and less
                # # Cov
                # cov = (advantages - advantages[response_mask].mean()) * (log_prob - log_prob[response_mask].mean()) * response_mask
                # # ratio
                # negative_approx_kl = log_prob - old_log_prob
                # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
                # ratio = torch.exp(negative_approx_kl)
                # if entropy is not None:
                #     response_mask, mask_entropy_less_than_threshold = self.get_top_entropy_mask_simple(entropy, response_mask, top_ratio=0.2)
                #     # lof metrics with different mask
                #     advantages_high_entropy = agg_loss(loss_mat=advantages, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                #     advantages_low_entropy = agg_loss(loss_mat=advantages, loss_mask=mask_entropy_less_than_threshold, loss_agg_mode=loss_agg_mode)
                #     entropy_high = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                #     entropy_low = agg_loss(loss_mat=entropy, loss_mask=mask_entropy_less_than_threshold, loss_agg_mode=loss_agg_mode)
                #     cov_high_entropy = agg_loss(loss_mat=cov, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                #     cov_low_entropy = agg_loss(loss_mat=cov, loss_mask=mask_entropy_less_than_threshold, loss_agg_mode=loss_agg_mode)
                #     ratio_high_entropy = agg_loss(loss_mat=ratio, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                #     ratio_low_entropy = agg_loss(loss_mat=ratio, loss_mask=mask_entropy_less_than_threshold, loss_agg_mode=loss_agg_mode)
                # else:
                #     advantages_high_entropy = torch.tensor(0.0, device=advantages_tmp.device)
                #     advantages_low_entropy = torch.tensor(0.0, device=advantages_tmp.device)
                #     entropy_high = torch.tensor(0.0, device=advantages_tmp.device)
                #     entropy_low = torch.tensor(0.0, device=advantages_tmp.device)
                #     cov_high_entropy = torch.tensor(0.0, device=advantages_tmp.device)
                #     cov_low_entropy = torch.tensor(0.0, device=advantages_tmp.device)
                #     ratio_high_entropy = torch.tensor(0.0, device=advantages_tmp.device)
                #     ratio_low_entropy = torch.tensor(0.0, device=advantages_tmp.device)
                    
                # pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                #                                                         log_prob=log_prob,
                #                                                         advantages=advantages,
                #                                                         eos_mask=response_mask,
                #                                                         clip_ratio_high=clip_ratio_high,
                #                                                         clip_ratio_low=clip_ratio_low)
                
                
                # # compute entropy loss from entropy
                # if entropy_coeff != 0:
                #     entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                #     # compute policy loss
                #     policy_loss = pg_loss - entropy_loss * entropy_coeff
                # else:
                #     policy_loss = pg_loss
                #     entropy_loss = torch.zeros_like(policy_loss)
                
                # loss = policy_loss / self.gradient_accumulation
                
                # loss.backward()
                
                # loss_info['actor/pg_loss'] = policy_loss.detach().item()
                # loss_info['actor/pg_clipfrac'] =  pg_clipfrac.detach().item()
                # loss_info['actor/ppo_kl'] = ppo_kl.detach().item()
                # loss_info['actor/entropy'] = entropy_loss.detach().item()
                # # modified: add metrics with different mask
                # loss_info['actor/advantages_high_entropy'] = advantages_high_entropy.detach().item()
                # loss_info['actor/advantages_low_entropy'] = advantages_low_entropy.detach().item()
                # loss_info['actor/entropy_high'] = entropy_high.detach().item()
                # loss_info['actor/entropy_low'] = entropy_low.detach().item()
                # loss_info['actor/cov_high_entropy'] = cov_high_entropy.detach().item()
                # loss_info['actor/cov_low_entropy'] = cov_low_entropy.detach().item()
                # loss_info['actor/ratio_high_entropy'] = ratio_high_entropy.detach().item()
                # loss_info['actor/ratio_low_entropy'] = ratio_low_entropy.detach().item()

                # append_to_dict(metrics, loss_info)
               
            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
            torch.cuda.empty_cache()
        self.actor_optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metrics

    
    def compute_entropy(self, bacth_data: DataProto):
        
        if bacth_data.meta_info['train_mode'] ==True:
            self.actor_module.train()
            print("train mode")
        else:
            self.actor_module.eval()
            print("eval mode")

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = bacth_data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values', "finish_step"]
        batch = bacth_data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size) # 8 * 64 / 128 = 4
        
        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1) *  responses.size(2)
                finish_step = data['finish_step'] * self.config.action_token_len
                steps = torch.arange(response_length, device=data['responses'].device)  # (traj_len,)
                steps_expanded = steps.unsqueeze(0).expand(data['responses'].size(0), -1)
                response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
                

                with torch.no_grad():
                    entropy = self._forward_micro_batch_entropy(micro_batch=data, temperature=temperature)
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                if bacth_data.meta_info['is_filtered'] and bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_after/entropy_loss_train': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif bacth_data.meta_info['is_filtered'] and not bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_after/entropy_loss_eval': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif not bacth_data.meta_info['is_filtered'] and bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_before/entropy_loss_train': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif not bacth_data.meta_info['is_filtered'] and not bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_before/entropy_loss_eval': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                        
                
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metrics