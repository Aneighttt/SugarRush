#!/bin/zsh

echo "================================"
echo "继续BC训练（额外100 epochs）"
echo "================================"
echo ""
echo "当前模型: bc_models_weighted/bc_policy_weighted.pt"
echo "额外训练: 100 epochs"
echo ""
read "proceed?是否继续训练? [y/n]: "

if [[ $proceed != "y" ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "🚀 启动继续训练..."
echo ""

source ~/Desktop/workspace/venv/bin/activate

python continue_bc_training.py \
    --checkpoint bc_models_weighted/bc_policy_weighted.pt \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.0003

echo ""
echo "================================"
echo "✅ 继续训练完成！"
echo "================================"
echo ""
echo "新模型已保存为:"
echo "  - bc_policy_weighted_continued.pt"
echo "  - bc_ppo_weighted_continued.zip"
echo ""
echo "如果满意新模型，运行以下命令替换旧模型:"
echo "  mv bc_models_weighted/bc_policy_weighted_continued.pt bc_models_weighted/bc_policy_weighted.pt"
echo "  mv bc_models_weighted/bc_ppo_weighted_continued.zip bc_models_weighted/bc_ppo_weighted.zip"
echo ""
