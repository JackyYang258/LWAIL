#!/bin/bash

# 环境列表
envs=(
  "ant-expert-v2"
  "halfcheetah-expert-v2"
  "hopper-expert-v2"
  "walker2d-expert-v2"
)

seeds=(0 1 2)

gpus=("cuda:0" "cuda:1" "cuda:2" "cuda:3" "cuda:4" "cuda:5" "cuda:6" "cuda:7")

max_training_timesteps=1000000
expert_episode="one"
downstream="td3"

mkdir -p logs

rounds=(
  # 格式: "update_timestep alpha wandb_name"
  "main"
)

# ================================
# 🚀 定义运行一轮的函数
# ================================
run_round() {
  local wandb_name=$1
  local gpu_idx=0

  echo "=========================================="
  echo "🚀 Starting round: ${wandb_name}"
  echo "=========================================="

  for env in "${envs[@]}"; do
    for seed in "${seeds[@]}"; do
      cuda=${gpus[$gpu_idx]}
      echo "===== Running ${env} seed ${seed} on ${cuda} ====="
      python main.py \
        --env_name ${env} \
        --f_epoch 30 \
        --max_training_timesteps ${max_training_timesteps} \
        --update_timestep ${4000} \
        --downstream ${downstream} \
        --expert_episode ${expert_episode} \
        --minus \
        --seed ${seed} \
        --using_icvf \
        --cuda ${cuda} \
        --wandb_name "${wandb_name}" \
        > logs/${env}_s${seed}_${wandb_name}.log 2>&1 &
      ((gpu_idx++))
    done
  done

  wait
  echo "✅ Round ${wandb_name} finished."
}

for round in "${rounds[@]}"; do
  run_round $round
done

echo "🎉 All rounds finished successfully!"
