# 🎯 PPO微调完整指南

## 📋 系统架构

```
BC训练 (已完成)
    ↓
bc_ppo_ready.zip (BC初始化的PPO)
    ↓
PPO微调训练 (train_ppo.py)
    ↓
ppo_finetuned.zip (微调后的PPO)
    ↓
robot.py (游戏中使用)
```

## 🚀 快速开始

### 1. 确认BC模型已就绪

```bash
ls -lh ./bc_models_imitation/bc_ppo_ready.zip
```

应该看到这个文件存在。

### 2. 启动PPO微调训练

```bash
python train_ppo.py \
    --timesteps 1000000 \
    --n_envs 4 \
    --lr 3e-4
```

### 3. 使用训练好的模型

训练完成后，`robot.py`会自动使用新模型（因为优先级设置）。

## ⚙️ 训练参数详解

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--timesteps` | 1,000,000 | 总训练步数 |
| `--n_envs` | 4 | 并行环境数量 |
| `--lr` | 3e-4 | 学习率 |
| `--n_steps` | 2048 | 每次更新的步数 |
| `--batch_size` | 64 | 批大小 |
| `--n_epochs` | 10 | PPO更新epochs |

### 训练时长估算

```
单环境:
- 1M steps ≈ 10-20小时 (取决于硬件)

4个并行环境:
- 1M steps ≈ 3-6小时

8个并行环境:
- 1M steps ≈ 2-4小时
```

### Callback参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save_freq` | 50,000 | Checkpoint保存频率 |
| `--eval_freq` | 25,000 | 评估频率 |

## 📊 监控训练

### 使用TensorBoard

```bash
tensorboard --logdir ./ppo_models/tensorboard
```

然后打开 http://localhost:6006

### 关键指标

- **ep_rew_mean**: 平均episode奖励（越高越好）
- **ep_len_mean**: 平均episode长度
- **approx_kl**: KL散度（应该较小）
- **loss**: 总损失（应该下降）
- **policy_loss**: 策略损失
- **value_loss**: 价值函数损失

## 🎮 robot.py模型加载逻辑

### 优先级

```python
1. ppo_finetuned.zip (PPO微调模型) ← 最高优先级
2. bc_ppo_ready.zip (BC初始化的PPO)
3. 随机AI (没有模型时)
```

### 配置选项

在`robot.py`开头：

```python
# 模型路径
PPO_FINETUNED_PATH = "./ppo_models/ppo_finetuned.zip"
PPO_BC_INIT_PATH = "./bc_models_imitation/bc_ppo_ready.zip"

# 开关
USE_PPO_MODEL = True  # True=使用PPO, False=随机AI
ENABLE_EXPERT_DATA_COLLECTION = False  # 是否收集数据
```

## 📈 训练阶段

### 阶段1：BC初始化（已完成）

```bash
python train_bc_imitation.py --n_epochs 50
# 输出: bc_ppo_ready.zip
```

### 阶段2：PPO微调（当前）

```bash
python train_ppo.py --timesteps 1000000
# 输出: ppo_finetuned.zip
```

### 阶段3：继续训练（可选）

```bash
python train_ppo.py \
    --continue_training \
    --checkpoint ./ppo_models/checkpoints/ppo_checkpoint_500000_steps.zip \
    --timesteps 1000000
```

## 💡 高级用法

### 1. 从头训练PPO（不用BC初始化）

```bash
# 删除或移动BC模型
mv ./bc_models_imitation/bc_ppo_ready.zip ./bc_models_imitation/bc_ppo_ready.zip.bak

# 训练
python train_ppo.py --timesteps 2000000
```

### 2. 调整学习率

```bash
# 更小的学习率（更稳定，但慢）
python train_ppo.py --lr 1e-4

# 更大的学习率（更快，但可能不稳定）
python train_ppo.py --lr 1e-3
```

### 3. 更多并行环境（需要更多内存）

```bash
python train_ppo.py --n_envs 8 --timesteps 2000000
```

### 4. 只训练短时间测试

```bash
python train_ppo.py --timesteps 50000 --n_envs 1
```

## 🎯 训练策略建议

### 快速测试（1小时）

```bash
python train_ppo.py \
    --timesteps 200000 \
    --n_envs 4 \
    --save_freq 25000
```

### 标准训练（3-6小时）

```bash
python train_ppo.py \
    --timesteps 1000000 \
    --n_envs 4 \
    --lr 3e-4
```

### 长时间训练（10+小时）

```bash
python train_ppo.py \
    --timesteps 5000000 \
    --n_envs 8 \
    --lr 1e-4
```

## 📁 输出文件结构

```
ppo_models/
├── ppo_finetuned.zip              ← 最终模型
├── checkpoints/
│   ├── ppo_checkpoint_50000_steps.zip
│   ├── ppo_checkpoint_100000_steps.zip
│   └── ...
├── eval/
│   ├── best_model.zip             ← 评估中最佳模型
│   └── evaluations.npz
└── tensorboard/
    └── ppo_run_1/
        └── events.out.tfevents.*
```

## 🔍 故障排除

### 问题1：内存不足

**症状**: 训练时内存占用过高

**解决**:
```bash
# 减少并行环境
python train_ppo.py --n_envs 2

# 或减少batch size
python train_ppo.py --batch_size 32
```

### 问题2：训练不收敛

**症状**: reward没有增长

**解决**:
1. 降低学习率: `--lr 1e-4`
2. 增加训练步数: `--timesteps 2000000`
3. 检查reward函数是否合理

### 问题3：BC模型找不到

**症状**: `未找到BC模型`

**解决**:
```bash
# 确认文件存在
ls ./bc_models_imitation/bc_ppo_ready.zip

# 或重新训练BC
python train_bc_imitation.py --n_epochs 50
```

## ✅ 完整工作流

### 从零开始到部署

```bash
# 1. 收集专家数据（已完成）
python robot.py  # with ENABLE_EXPERT_DATA_COLLECTION=True

# 2. BC训练（已完成）
python train_bc_imitation.py --n_epochs 50

# 3. PPO微调（当前步骤）
python train_ppo.py --timesteps 1000000 --n_envs 4

# 4. 监控训练
tensorboard --logdir ./ppo_models/tensorboard

# 5. 测试模型
python robot.py  # with USE_PPO_MODEL=True

# 6. 继续优化（可选）
python train_ppo.py \
    --continue_training \
    --checkpoint ./ppo_models/ppo_finetuned.zip \
    --timesteps 1000000
```

## 📝 总结

### 当前状态
- ✅ BC训练完成
- ✅ bc_ppo_ready.zip已生成
- ✅ robot.py支持PPO预测
- ⏳ **现在可以开始PPO微调**

### 下一步
```bash
# 启动PPO训练
python train_ppo.py
```

### 预期结果
- 训练3-6小时
- 获得ppo_finetuned.zip
- 性能超越BC（有可能超越专家）

---

**开始你的PPO微调之旅吧！** 🚀

