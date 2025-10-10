#!/bin/zsh

echo "================================"
echo "BC加权训练 - 提升关键动作权重"
echo "================================"
echo ""
echo "⭐ 改进点:"
echo "  - 放炸弹权重 x20"
echo "  - 减速动作权重 x5-8"
echo "  - 独立监控各子动作准确率"
echo ""
echo "预期效果:"
echo "  - bomb_acc: 0.4 → 0.7+"
echo "  - speed_acc: 0.5 → 0.7+"
echo "  - all_correct: 0.48 → 0.58-0.65"
echo ""
read "proceed?是否开始训练? [y/n]: "

if [[ $proceed != "y" ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "🚀 启动训练..."
echo ""

source ~/Desktop/workspace/venv/bin/activate

python train_bc_weighted.py \
    --n_epochs 1000 \
    --batch_size 128 \
    --lr 0.0002

echo ""
echo "================================"
echo "✅ 训练完成！"
echo "================================"
