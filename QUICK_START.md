# 快速开始加权BC训练

## 问题已修复 ✅

原错误: `TypeError: 'Linear' object is not subscriptable`
修复方案: 使用 imitation 库 + 继承 BC 类重写损失函数

## 立即开始训练

```bash
./TRAIN_WEIGHTED.sh
```

或直接运行:
```bash
source ~/Desktop/workspace/venv/bin/activate
python train_bc_weighted.py --n_epochs 150 --batch_size 128 --lr 0.0003
```

## 技术方案

- ✅ 基于 imitation 库（稳定可靠）
- ✅ 自定义加权损失（提升关键动作）
- ✅ 100%兼容 PPO（使用 ActorCriticPolicy）
- ✅ 自动转换为 PPO 格式

## 预期效果

- prob_true_act: 0.48 → 0.58-0.65
- bomb_acc: 提升到 0.70+
- speed_acc: 提升到 0.70+

训练时间: 约2-2.5小时
