set -x
export PYTHONUNBUFFERED=1

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export USED_MODEL="${USED_MODEL:-no_api}"

# Run this script from the IW-OPD/verl directory:
#   cd IW-OPD/verl
#   bash examples/opd/run_qwen3-30b-a3b-instruct-opd_4b_iw_opd.sh
#
# Expected local layout:
#   IW-OPD/models/Qwen3-4B
#   IW-OPD/models/Qwen3-30B-A3B-Instruct-2507
#   IW-OPD/data/DeepMath-103K/train_filtered_level6.parquet
#   IW-OPD/data/AIME2024/test.parquet
#   IW-OPD/data/AIME2025/test.parquet

DATA_ROOT="${DATA_ROOT:-../data}"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-../models/Qwen3-4B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-../models/Qwen3-30B-A3B-Instruct-2507}"
OUTPUT_DIR="${OUTPUT_DIR:-../checkpoints/Qwen3-4B-Teacher-Qwen3-30B-A3B-Instruct-2507-IW-OPD}"
IW_OPD_WEIGHT_ENABLE="${IW_OPD_WEIGHT_ENABLE:-true}"
IW_OPD_WEIGHT_MAX="${IW_OPD_WEIGHT_MAX:-1.5}"

aime24_test_path="${DATA_ROOT}/AIME2024/test.parquet"
aime25_test_path="${DATA_ROOT}/AIME2025/test.parquet"

test_files="['$aime24_test_path', '$aime25_test_path']"

# IW-OPD keeps the standard OPD advantage direction and applies a
# stop-gradient cumulative-disagreement weight:
#   d_t = |log p_teacher(y_t|prefix) - log p_student(y_t|prefix)|
#   F_(t-1) = sum_{k<t} d_k / sum_{k<=T} d_k
#   g_t = 1 + (weight_max - 1) * (1 - F_(t-1))
python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.rollout_correction.rollout_is=token \
        algorithm.rollout_correction.rollout_is_threshold=5.0 \
        algorithm.rollout_correction.rollout_rs=null \
        algorithm.rollout_correction.bypass_mode=false \
        actor_rollout_ref.rollout.calculate_log_probs=true \
        data.train_files="${DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet" \
        data.val_files="$test_files" \
        data.train_batch_size=1024 \
        data.max_prompt_length=2048 \
        data.max_response_length=16384 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=True \
        data.seed=42 \
        data.return_raw_chat=True \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        actor_rollout_ref.model.path="$STUDENT_MODEL_PATH" \
        +actor_rollout_ref.ref.model.path="$TEACHER_MODEL_PATH" \
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
        actor_rollout_ref.actor.policy_loss.lambda_vals=1.0 \
        actor_rollout_ref.actor.policy_loss.iw_opd_weight_enable="$IW_OPD_WEIGHT_ENABLE" \
        actor_rollout_ref.actor.policy_loss.iw_opd_weight_max="$IW_OPD_WEIGHT_MAX" \
        actor_rollout_ref.actor.policy_loss.iw_opd_weight_use_abs=true \
        actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.n=32 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        reward_model.reward_manager=naive \
        trainer.critic_warmup=0 \
        trainer.val_before_train=True \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=10 \
        trainer.project_name='iw-opd' \
        trainer.experiment_name='qwen3_4b_teacher_qwen3_30b_a3b_instruct_iw_opd' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=4 \
        trainer.save_freq=50 \
        trainer.default_local_dir="$OUTPUT_DIR" \
        trainer.test_freq=10 \
        trainer.total_epochs=3 "$@"
