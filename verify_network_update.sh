#!/bin/zsh

echo "========================================="
echo "验证网络结构升级"
echo "========================================="
echo ""

success=0
total=0

# 检查 train_bc_weighted.py
echo "🔍 检查 train_bc_weighted.py..."
total=$((total + 1))
if grep -q "net_arch=dict(pi=\[256, 128, 64\], vf=\[256, 128, 64\])" train_bc_weighted.py; then
    echo "  ✅ 已更新为 [256, 128, 64]"
    success=$((success + 1))
else
    echo "  ❌ 未更新或格式不正确"
fi

# 检查 train_bc_imitation.py
echo ""
echo "🔍 检查 train_bc_imitation.py..."
total=$((total + 1))
if grep -q "net_arch=dict(pi=\[256, 128, 64\], vf=\[256, 128, 64\])" train_bc_imitation.py; then
    echo "  ✅ 已更新为 [256, 128, 64]"
    success=$((success + 1))
else
    echo "  ❌ 未更新或格式不正确"
fi

# 检查 train_ppo.py
echo ""
echo "🔍 检查 train_ppo.py..."
total=$((total + 1))
if grep -q "net_arch=dict(pi=\[256, 128, 64\], vf=\[256, 128, 64\])" train_ppo.py; then
    echo "  ✅ 已更新为 [256, 128, 64]"
    success=$((success + 1))
else
    echo "  ❌ 未更新或格式不正确"
fi

# 检查旧模型是否存在
echo ""
echo "🔍 检查旧模型..."
total=$((total + 1))
if [ -d "bc_models_imitation" ] || [ -d "bc_models_weighted" ] || [ -d "ppo_models" ]; then
    echo "  ⚠️  检测到旧模型目录，建议删除："
    [ -d "bc_models_imitation" ] && echo "     - bc_models_imitation/"
    [ -d "bc_models_weighted" ] && echo "     - bc_models_weighted/"
    [ -d "ppo_models" ] && echo "     - ppo_models/"
    echo ""
    echo "  删除命令:"
    echo "     rm -rf bc_models_imitation/ bc_models_weighted/ ppo_models/"
else
    echo "  ✅ 无旧模型（或已删除）"
    success=$((success + 1))
fi

# 总结
echo ""
echo "========================================="
echo "验证结果: $success/$total 通过"
echo "========================================="

if [ $success -eq $total ]; then
    echo ""
    echo "🎉 所有检查通过！可以开始训练了："
    echo ""
    echo "   ./TRAIN_WEIGHTED.sh"
    echo ""
else
    echo ""
    echo "⚠️  有 $((total - success)) 项未通过，请检查上述问题"
    echo ""
fi
