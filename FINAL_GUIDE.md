# 🎓 完整训练指南

## ✅ 已完成

1. ✅ **数据收集系统** - 150局专家数据已收集
2. ✅ **imitation库集成** - BC训练器已就绪
3. ✅ **自定义BC训练器** - 备用方案
4. ✅ **代码清理** - 删除冗余文件

## 📁 精简后的项目结构

```
SugarRush/
├── 🎮 核心组件
│   ├── robot.py                      # Flask服务器
│   ├── environment.py                # Gym环境
│   ├── config.py                     # 配置
│   ├── data_models.py               # 数据结构
│   └── frame_processor.py           # 帧处理
│
├── 📊 数据收集
│   ├── realtime_expert_collector.py # 实时收集器
│   ├── bc_data_collector.py         # 数据工具
│   └── expert_data/                 # 数据目录
│       ├── expert_1/ (150 episodes)
│       └── expert_3/ (150 episodes)
│
├── 🤖 BC训练
│   ├── train_bc_imitation.py        # imitation库训练（推荐）
│   ├── bc_trainer.py                # 自定义训练器
│   └── START_TRAINING.sh            # 快速启动脚本
│
├── 📖 文档
│   ├── README.md                    # 原项目文档
│   ├── README_SIMPLE.md            # 简化版文档
│   ├── TRAINING_QUICKSTART.md      # 快速训练指南
│   ├── TRAINING_GUIDE.md           # 详细指南
│   ├── AUTO_COLLECTION_GUIDE.md    # 数据收集说明
│   └── QUICK_START.md              # 快速开始
│
└── 🛠️ 工具
    ├── action_converter.py          # 动作转换
    ├── utils.py                     # 工具函数
    └── test_realtime_collector.py  # 测试
```

## 🚀 现在开始训练

### 方法1：使用启动脚本（最简单）

```bash
./START_TRAINING.sh
```

### 方法2：直接运行Python

```bash
python train_bc_imitation.py
```

### 方法3：自定义参数

```bash
python train_bc_imitation.py \
    --n_epochs 100 \
    --batch_size 32 \
    --lr 0.0005 \
    --evaluate
```

## 📈 训练流程

```
数据收集 (已完成)
    ↓
BC训练 (现在)
    ├─ 使用imitation库 (推荐)
    └─ 或自定义训练器
    ↓
评估模型
    ├─ 验证准确率
    └─ 游戏中测试
    ↓
PPO微调 (可选)
    └─ 超越专家水平
```

## 🎯 训练参数建议

| 数据量 | Epochs | Batch Size | Learning Rate |
|--------|--------|------------|---------------|
| 100-150局 | 30-50 | 64 | 0.001 |
| 150-200局 | 50-80 | 64 | 0.001 |
| 200+局 | 80-100 | 128 | 0.0005 |

## 🔍 关键代码说明

### imitation库的优势

```python
# 自动处理MultiDiscrete动作空间
from imitation.algorithms import bc

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,  # MultiDiscrete([5, 2, 5])
    demonstrations=trajectories,
    device='cuda'
)

bc_trainer.train(n_epochs=50)
```

### 无缝对接PPO

```python
# BC训练后直接转换为PPO格式
ppo_model = PPO(...)
ppo_model.policy.load_state_dict(bc_trainer.policy.state_dict())
ppo_model.save("bc_ppo_ready.zip")

# 之后可以直接用于PPO微调
model = PPO.load("bc_ppo_ready.zip")
model.learn(total_timesteps=1000000)
```

## ⚙️ 环境配置

### GPU推荐配置

```bash
# CUDA 11.8+
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### CPU配置（较慢）

```bash
pip install torch
pip install -r requirements.txt
```

## 📊 监控训练

### TensorBoard（如果启用）

```bash
tensorboard --logdir ./logs
```

### 训练输出示例

```
找到 2 个专家: expert_1, expert_3
✅ 总样本数: 540,000

Epoch 1/50: Loss=1.234, Acc=65.4%
Epoch 2/50: Loss=1.123, Acc=68.2%
...
Epoch 50/50: Loss=0.789, Acc=82.1%

✅ 训练完成！
✅ BC策略: ./bc_models_imitation/bc_policy
✅ PPO格式: ./bc_models_imitation/bc_ppo_ready.zip
```

## 🎮 使用训练好的模型

### 在robot.py中使用

```python
from imitation.algorithms import bc

# 加载BC策略
policy = bc.reconstruct_policy("./bc_models_imitation/bc_policy")

# 在handle_command中预测
action, _ = policy.predict(observation)
```

### 独立测试

```python
from environment import BomberEnv
from imitation.algorithms import bc

env = BomberEnv()
policy = bc.reconstruct_policy("./bc_models_imitation/bc_policy")

obs = env.reset()
for _ in range(1000):
    action, _ = policy.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
```

## 📝 下一步计划

### 短期（BC训练后）
1. 评估BC模型性能
2. 在游戏中测试
3. 收集更多数据（如果需要）

### 中期（PPO微调）
1. 使用BC模型初始化PPO
2. 自我博弈训练
3. 调整奖励函数

### 长期（优化提升）
1. 超参数调优
2. 网络结构优化
3. 多模型集成

## ❓ 常见问题

### Q1: imitation库和自定义trainer有什么区别？

| 特性 | imitation | 自定义 |
|-----|-----------|--------|
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| SB3集成 | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 灵活性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| PPO对接 | ⭐⭐⭐⭐⭐ | ⭐⭐ |

**推荐使用imitation库** - 除非有特殊需求

### Q2: 训练多久合适？

- **最少**: 30 epochs (~30分钟，GPU)
- **推荐**: 50 epochs (~1小时，GPU)
- **充分**: 100 epochs (~2小时，GPU)

### Q3: 如何知道训练效果好？

**指标检查**:
- ✅ 验证准确率 > 65%
- ✅ 损失下降并收敛
- ✅ 游戏中表现明显优于随机

**实际测试**:
- 加载模型到robot.py
- 对战测试
- 观察存活时间和击杀数

## 🎉 总结

### 已准备就绪
- ✅ 150局高质量数据
- ✅ imitation库BC训练器
- ✅ 完整的训练流程
- ✅ PPO微调接口

### 现在就开始

```bash
# 一键启动
./START_TRAINING.sh

# 或
python train_bc_imitation.py
```

---

**祝训练顺利！有问题随时查看文档或源码注释。** 🚀

