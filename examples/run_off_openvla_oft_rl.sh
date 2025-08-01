set -x

export NCCL_DEBUG=WARN 
export SWANLAB_API_KEY=XLd02pP97wfHTqZfSWo0Y          # 设置在线跟踪模式API
# export SWANLAB_LOG_DIR=<设置本地日志存储路径>    # 设置本地日志存储路径
# export SWANLAB_MODE=<设置SwanLab的运行模式>     # 包含四种模式：cloud云端跟踪模式（默认）、cloud-only仅云端跟踪本地不保存文件、local本地跟踪模式、disabled完全不记录用于debug
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# For openvla-oft Libero-Long traj1 SFT or traj all SFT models can be find in https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86
SFT_MODEL_PATH="/share/home/u16023/jinyiyang/codebase/checkpoints/Openvla-oft-SFT-libero10-trajall"
CKPT_PATH="/share/home/u16023/jinyiyang/experiments/SimpleVLA-RL/checkpoints"
# DATASET_NAME can be libero_10 (libero_Long), libero_90, libero_spatial, libero_object, libero_goal
DATASET_NAME="libero_10"
VLA_NAME="openvla-oft"
NUM_GPUS=8
# If you want to use 2*8 GPU to RL. Set NUM_NODES=2
NUM_NODES=1 
ALIGN_PATH="/share/home/u16023/jinyiyang/experiments/SimpleVLA-RL/align.json"


# train setting
n_samples=8
train_batch_size=32
ppo_mini_batch_size=64
traj_mini_batch_size=16

# algo setting
accuracy_lower_bound=0.1
accuracy_upper_bound=0.9

clip_ratio_high=0.5
clip_ratio_low=0.3

relabel=False
relabel_step=20
max_version_diff=1
high_adv_threshold=1.0
entropy_coeff=0.0
on_data_ratio=0.5
off_policy_reshape=p_div_p_0.1 # no_reshape/ logp / p_logp / square_root / p_div_p_0.1 / p_div_p_0.3 / p_div_p_0.5 / pow

# logger="['console']"
logger="['console','swanlab']"

PROJECT_NAME=SimpleVLA-RL
EXPERIMENT_NAME=offline-${VLA_NAME}-${DATASET_NAME}-grpo_n${n_samples}_b${train_batch_size}_mb${ppo_mini_batch_size}_t${traj_mini_batch_size}_acc${accuracy_lower_bound}_${accuracy_upper_bound}_clip${clip_ratio_low}_${clip_ratio_high}_relabel-${relabel}-${relabel_step}_mv${max_version_diff}_adv${high_adv_threshold}_ety${entropy_coeff}_on${on_data_ratio}_${off_policy_reshape} #'test-buffer-online_a100_n8_b32_woentropy_cl0.28_acc0.1_0.9' 

HYDRA_FULL_ERROR=1 python -m verl.trainer.main_ppo \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=50 \
    data.n_samples=${n_samples} \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=${accuracy_lower_bound} \
    data.accuracy_upper_bound=${accuracy_upper_bound} \
    data.oversample_factor=1 \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=496 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    data.max_buffer_size=4096 \
    data.min_buffer_size=128 \
    data.max_version_diff=${max_version_diff} \
    data.relabel=${relabel} \
    data.relabel_step=${relabel_step} \
    data.on_data_ratio=${on_data_ratio} \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=${traj_mini_batch_size} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.val_micro_batch_size=8 \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.unnorm_key=$DATASET_NAME \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    algorithm.high_adv_threshold=${high_adv_threshold} \
    trainer.logger=${logger} \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    trainer.val_only=False \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.val_before_train=False \
    actor_rollout_ref.actor.use_sft_prefix_reward=False \
    actor_rollout_ref.actor.use_off_policy_loss=True \
    actor_rollout_ref.actor.off_policy_normalize=False \
    actor_rollout_ref.actor.off_policy_reshape=${off_policy_reshape} \
    actor_rollout_ref.actor.off_policy_loss_impl=token \
    actor_rollout_ref.actor.loss_remove_token_mean=False \
    actor_rollout_ref.actor.loss_remove_clip=False \
    
# 2>&1 | tee -a train_log_l40_test.txt


