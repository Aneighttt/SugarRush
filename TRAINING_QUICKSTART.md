# 🚀 训练快速开始

## 前置条件

✅ 已收集150局专家数据  
✅ 数据在 `./expert_data/expert_1/` 和 `./expert_data/expert_3/`

## 方法1：使用imitation库（推荐）

### 优势
- ✅ 与Stable Baselines3无缝集成
- ✅ 可直接用于PPO微调
- ✅ 成熟稳定
- ✅ 支持MultiDiscrete动作空间

### 训练命令

```bash
python train_bc_imitation.py \
    --data_dir ./expert_data \
    --output_dir ./bc_models_imitation \
    --n_epochs 50 \
    --batch_size 64 \
    --lr 0.001
```

### 训练输出

```
bc_models_imitation/
├── bc_policy/              ← BC策略（imitation格式）
└── bc_ppo_ready.zip       ← PPO格式（可直接微调）
```

### 使用训练好的模型

```python
from imitation.algorithms import bc

# 加载BC策略
policy = bc.reconstruct_policy("./bc_models_imitation/bc_policy")

# 预测
action, _ = policy.predict(observation)
```

### PPO微调

```python
from stable_baselines3 import PPO

# 加载BC初始化的PPO模型
model = PPO.load("./bc_models_imitation/bc_ppo_ready.zip")

# 继续训练
model.learn(total_timesteps=1000000)
```

## 方法2：使用自定义BC训练器

### 特点
- 🎯 完全自定义
- 📊 详细训练曲线
- 🔧 灵活的网络结构

### 使用方法

```python
from bc_trainer import BCPolicyNetwork, BCTrainer, ExpertDataset
from bc_data_collector import ExpertDataLoader
from torch.utils.data import DataLoader

# 1. 加载数据
loader = ExpertDataLoader("./expert_data/expert_1")
loader.load_data()
obs_dict, actions = loader.get_transitions_as_arrays()

# 2. 创建数据集
dataset = ExpertDataset(obs_dict, actions)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 3. 创建模型
model = BCPolicyNetwork(
    grid_channels=14,
    grid_height=16,
    grid_width=28,
    player_state_dim=10
)

# 4. 训练
trainer = BCTrainer(model, device='cuda')
trainer.train(dataloader, num_epochs=50)
```

## 训练时间估计

| 硬件配置 | 50 epochs训练时间 |
|---------|------------------|
| CPU | ~12小时 |
| GPU (GTX 1080) | ~2小时 |
| GPU (RTX 3080) | ~1小时 |

## 期望结果

### 良好的训练指标

- **训练准确率**: 70-85%
- **验证准确率**: 65-80%
- **损失收敛**: Loss < 1.0

### 各维度准确率

| 动作维度 | 期望准确率 | 说明 |
|---------|-----------|------|
| 方向 | 60-70% | 最难预测 |
| 炸弹 | 85-95% | 较简单 |
| 速度 | 90-95% | 最简单 |

## 常见问题

### Q1: 显存不足

```bash
# 减小batch size
python train_bc_imitation.py --batch_size 32
```

### Q2: 准确率不高

- 增加训练轮数: `--n_epochs 100`
- 收集更多数据
- 检查数据质量

### Q3: 训练太慢

- 使用GPU
- 增加 `num_workers` (DataLoader)
- 使用更小的模型

## 下一步

训练完成后：

1. **评估模型**
   ```bash
   python train_bc_imitation.py --evaluate
   ```

2. **在游戏中使用**
   - 修改 `robot.py` 加载BC模型
   - 启动游戏测试

3. **PPO微调**
   ```bash
   python train_ppo.py --bc_model ./bc_models_imitation/bc_ppo_ready.zip
   ```

---

**现在就开始训练吧！** 🎉

```bash
python train_bc_imitation.py
```

