#!/usr/bin/env bash
set -euxo pipefail

project_name='agentic_rl_alfworld'
exp_name='proxmo_qwen2.5_7b'

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/.cache/model_data"}

export HF_HOME=${RAY_DATA_HOME}/.cache/huggingface
export HF_DATASETS_CACHE=${RAY_DATA_HOME}/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=${RAY_DATA_HOME}/.cache/huggingface/transformers

MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-7B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpt/${project_name}/${exp_name}"}

train_data_size=16
val_data_size=128

TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/agent/${train_data_size}_${val_data_size}/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/agent/${train_data_size}_${val_data_size}/test.parquet"}

group_size=8  
mode="soft_grouping"
num_cpus_per_env_worker=0.1

max_prompt_length=2048  
max_response_length=512

n_gpus_per_node=8
NNODES=${NNODES:-1}

total_epochs=150  
save_freq=25  
test_freq=5
val_before_train=False

# 推理配置
temperature=0.4
do_sample=True
tensor_model_parallel_size=2
gpu_memory_utilization=0.8

ppo_mini_batch_size=256
ppo_micro_batch_size_per_gpu=32
log_prob_micro_batch_size_per_gpu=32

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=proxmo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${train_data_size} \
    data.val_batch_size=${val_data_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${do_sample} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.proxmo.step_advantage_w=1.0 \
    algorithm.proxmo.mode=${mode} \
    +algorithm.proxmo.enable_psc=True \
    +algorithm.proxmo.psc_alpha=4.0 \
    +algorithm.proxmo.psc_target_range=0.05 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=${group_size} \
    env.resources_per_worker.num_cpus=${num_cpus_per_env_worker} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.val_before_train=${val_before_train} $@