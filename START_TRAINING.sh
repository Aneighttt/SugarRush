#!/bin/bash

# BC训练启动脚本

echo "======================================"
echo "  炸弹人AI - BC训练"
echo "======================================"
echo ""

# 检查数据目录
if [ ! -d "./expert_data" ]; then
    echo "❌ 错误: 找不到 expert_data 目录"
    echo "   请先运行 robot.py 收集数据"
    exit 1
fi

# 统计数据量
expert_count=$(ls -d ./expert_data/expert_* 2>/dev/null | wc -l)
if [ $expert_count -eq 0 ]; then
    echo "❌ 错误: expert_data 目录为空"
    echo "   请先运行 robot.py 收集数据"
    exit 1
fi

echo "✅ 找到 $expert_count 个专家数据"
echo ""

# 询问训练方法
echo "选择训练方法:"
echo "  1) imitation库 (推荐)"
echo "  2) 自定义BC训练器"
echo -n "请选择 (1/2): "
read choice

case $choice in
    1)
        echo ""
        echo "使用 imitation 库训练..."
        python train_bc_imitation.py \
            --data_dir ./expert_data \
            --output_dir ./bc_models_imitation \
            --n_epochs 100 \
            --batch_size 128 \
            --lr 0.0005
        ;;
    2)
        echo ""
        echo "使用自定义BC训练器..."
        echo "请直接运行 Python 脚本或参考 bc_trainer.py"
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "  训练完成！"
echo "======================================"

