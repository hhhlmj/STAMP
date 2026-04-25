#!/bin/bash

# TG-GEN 模型训练脚本
# 使用方式: ./train.sh

echo "=========================================="
echo "🚀 TG-GEN 模型训练启动"
echo "=========================================="

python3 main.py \
    -d ICEWS14s \
    --train-history-len 3 \
    --test-history-len 3 \
    --dilate-len 1 \
    --lr 0.001 \
    --n-layers 2 \
    --evaluate-every 1 \
    --gpu 0 \
    --n-hidden 200 \
    --self-loop \
    --decoder convtranse \
    --encoder uvrgcn \
    --layer-norm \
    --weight 0.5 \
    --entity-prediction \
    --relation-prediction \
    --add-static-graph \
    --angle 10 \
    --discount 1 \
    --task-weight 0.7 \
    --twin-period 365 \
    --early-stopping-patience 20

echo ""
echo "✅ 训练完成！"
echo "📊 查看日志: cd .. && python3 analyze_training_logs.py --plot"
