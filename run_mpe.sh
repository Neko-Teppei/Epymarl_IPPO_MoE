#!/usr/bin/env bash
set -e # 遇到错误立即退出

# ========================================================
# 实验超参数设置
# ========================================================
# 至少 3 个种子以保证统计学意义 (消除偶然性)
SEEDS=(45 46 47)

# 你的算法与基线大乱斗
# ippo: 信息最少的下限
# ippo_moe: 你的算法
# mappo: 信息最多的上限 (CTDE)
ALGOS=(ippo_moe ippo mappo)

# 原生 MPE 任务名称 (可换成 mpe:SimpleSpeakerListener-v3 等)
MAP_NAME="mpe:SimpleSpread-v3"

# 真实训练步数 (建议跑大实验时设为 2000000，目前测试设为 100000)
T_MAX=100000

# ========================================================
# 自动化执行循环
# ========================================================
for algo in "${ALGOS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "========================================================"
    echo "🚀 Starting: Algo=${algo} | Seed=${seed} | Env=${MAP_NAME}"
    echo "========================================================"

    # 核心改动：使用 gymma 启动原生环境，并通过 with 传入覆盖参数
    python src/main.py \
      --config="$algo" \
      --env-config=gymma \
      with \
      env_args.time_limit=50 \
      env_args.key="$MAP_NAME" \
      seed="$seed" \
      t_max="$T_MAX"
  done
done

echo "🎉 所有实验全部运行完毕！"
