# SugarRush BC+PPO 训练指南

本指南介绍如何使用行为克隆（BC）和PPO强化学习训练炸弹人AI。

## 📋 目录

1. [环境准备](#环境准备)
2. [训练流程](#训练流程)
3. [方案对比](#方案对比)
4. [使用方法](#使用方法)
5. [模型架构](#模型架构)
6. [常见问题](#常见问题)

---

## 🔧 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：
- `stable-baselines3`: 强化学习算法库（PPO）
- `imitation`: 模仿学习算法库（BC）
- `gymnasium`: OpenAI Gym环境
- `torch`: PyTorch深度学习框架
- `numpy`: 数值计算

### 2. requirements.txt

创建`requirements.txt`文件：

```txt
gymnasium>=0.29.0
stable-baselines3>=2.0.0
imitation>=1.0.0
torch>=2.0.0
numpy>=1.24.0
tensorboard>=2.14.0
```

安装：
```bash
pip install -r requirements.txt
```

---

## 🎯 训练流程

### 训练流程图

```
专家AI游戏数据
      ↓
   数据收集
      ↓
   专家数据集
      ↓
   BC训练 (imitation库)
      ↓
   BC预训练模型
      ↓
   PPO微调 (stable-baselines3)
      ↓
   最终模型
```

---

## 📊 方案对比

### 方案1: 离线训练（推荐）

**流程**：
1. 先收集专家数据并保存到本地
2. 从本地数据训练BC模型
3. 使用PPO微调BC模型

**优点**：
- ✅ 数据可重复使用
- ✅ 训练稳定，可随时中断恢复
- ✅ 便于数据分析和清洗
- ✅ 可以混合多次游戏的数据

**缺点**：
- ❌ 需要额外的数据收集步骤
- ❌ 占用存储空间

**适用场景**：
- 专家AI质量高，数据价值大
- 需要多次实验调参
- 计算资源有限，需要分步训练

### 方案2: 在线训练

**流程**：
1. 实时收集专家数据并立即训练
2. 边玩边学

**优点**：
- ✅ 实时反馈
- ✅ 无需存储大量数据
- ✅ 可以持续学习

**缺点**：
- ❌ 训练不稳定
- ❌ 数据不可复用
- ❌ 中断后难以恢复

**适用场景**：
- 快速原型验证
- 实时学习场景

---

## 📖 使用方法

### 步骤1: 收集专家数据

```bash
python collect_expert_data.py \
    --episodes 100 \
    --save_dir ./expert_data \
    --save_interval 10
```

**参数说明**：
- `--episodes`: 收集的游戏局数
- `--save_dir`: 数据保存目录
- `--save_interval`: 每隔多少局保存一次

**注意**: 你需要修改`collect_expert_data.py`中的数据收集逻辑，确保正确获取专家AI的动作。

### 步骤2: BC训练

#### 方式A: 单独训练BC

```bash
python train_bc_ppo.py \
    --mode bc_only \
    --expert_data ./expert_data \
    --bc_epochs 100 \
    --bc_batch_size 128 \
    --bc_save_path ./models/bc_policy.zip \
    --device auto \
    --evaluate
```

**参数说明**：
- `--mode bc_only`: 仅训练BC
- `--expert_data`: 专家数据路径
- `--bc_epochs`: BC训练轮数
- `--bc_batch_size`: 批次大小
- `--bc_save_path`: 模型保存路径
- `--device`: 训练设备 (cpu/cuda/auto)
- `--evaluate`: 训练后评估模型

### 步骤3: PPO微调

```bash
python train_bc_ppo.py \
    --mode ppo_only \
    --bc_save_path ./models/bc_policy.zip \
    --ppo_timesteps 1000000 \
    --ppo_save_path ./models/ppo_finetuned.zip \
    --ppo_log_dir ./logs/ppo \
    --n_envs 4 \
    --device auto \
    --evaluate
```

**参数说明**：
- `--mode ppo_only`: 仅PPO微调
- `--bc_save_path`: BC模型路径（用于初始化）
- `--ppo_timesteps`: PPO训练总步数
- `--ppo_save_path`: PPO模型保存路径
- `--ppo_log_dir`: TensorBoard日志目录
- `--n_envs`: 并行环境数（建议2-8）
- `--evaluate`: 训练后评估模型

### 步骤4: 完整流程（一键训练）

```bash
python train_bc_ppo.py \
    --mode all \
    --expert_data ./expert_data \
    --bc_epochs 100 \
    --ppo_timesteps 1000000 \
    --device auto \
    --evaluate
```

**参数说明**：
- `--mode all`: 完整流程（BC + PPO）

### 查看训练日志

```bash
tensorboard --logdir ./logs/ppo
```

然后在浏览器打开 `http://localhost:6006`

---

## 🏗️ 模型架构

### 观察空间

**Dict类型观察**：
```python
{
    "grid_view": Box(shape=(13, 16, 28), dtype=float32),
    "player_state": Box(shape=(8,), dtype=float32)
}
```

**grid_view通道说明**（13个通道）：
- Channel 0: 地形（0=可通过, 0.5=软墙, 1.0=硬墙）
- Channel 1: 炸弹位置
- Channel 2: 危险区域（爆炸范围）
- Channel 3-5: 道具（鞋子、药水、炸药包）
- Channel 6-7: 特殊地形（加速、减速）
- Channel 8-9: 占领区域（敌方、中立）
- Channel 10: 路径梯度
- Channel 11: 我方玩家
- Channel 12: 队友

**player_state说明**（8维向量）：
1. offset_x: 玩家在格子内的x偏移
2. offset_y: 玩家在格子内的y偏移
3. bomb_pack_count: 炸弹包数量（归一化）
4. bomb_range: 炸弹范围（归一化）
5. speed: 移动速度（归一化）
6. can_place_bomb: 是否可以放炸弹
7. is_stunned: 是否眩晕
8. is_invincible: 是否无敌

### 动作空间

**Discrete(6)**：
- 0: 上
- 1: 下
- 2: 左
- 3: 右
- 4: 放炸弹
- 5: 停止

### BC网络架构

使用**imitation库**的BC实现，基于SB3的MultiInputPolicy：

```python
策略网络结构:
- Features Extractor (CNN for grid_view):
  - Conv2d(13, 32, kernel_size=3, stride=1, padding=1)
  - Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
  - Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
  
- MLP for player_state:
  - Linear(8, 64)
  - Linear(64, 64)
  
- Policy Head:
  - Linear(combined_features, 256)
  - Linear(256, 6)  # 6个动作
```

### PPO网络架构

继承BC网络权重，增加价值函数：

```python
策略网络 (Actor):
- 继承自BC网络
- 输出动作概率分布

价值网络 (Critic):
- 共享Features Extractor
- 独立的MLP头
- 输出状态价值V(s)
```

---

## ❓ 常见问题

### Q1: 地图坐标系统

**问题**: 地图是28宽×16高，如何在numpy中表示？

**回答**: 
- 游戏坐标: (width=28, height=16)
- numpy维度: `(channels, height, width) = (13, 16, 28)`
- 访问方式: `grid_view[channel, y, x]`
- 注意: y是行（高度），x是列（宽度）

### Q2: BC训练准确率低

**可能原因**：
1. 专家数据质量不高
2. 专家数据量不足（建议至少50-100局）
3. 训练轮数不够

**解决方案**：
```bash
# 增加训练轮数
--bc_epochs 200

# 减小批次大小
--bc_batch_size 64

# 增加专家数据
python collect_expert_data.py --episodes 200
```

### Q3: PPO训练不收敛

**可能原因**：
1. 学习率过高
2. BC预训练不充分
3. 奖励函数设计不合理

**解决方案**：
```bash
# 降低学习率
--ppo_lr 1e-4

# 增加训练步数
--ppo_timesteps 2000000

# 调整PPO参数
--ppo_n_steps 4096
--ppo_batch_size 128
```

### Q4: 内存不足

**解决方案**：
```bash
# 减小批次大小
--bc_batch_size 64
--ppo_batch_size 32

# 减少并行环境
--n_envs 1

# 使用CPU训练
--device cpu
```

### Q5: 如何使用训练好的模型

```python
from stable_baselines3 import PPO
from environment import BomberEnv

# 加载模型
model = PPO.load("./models/ppo_finetuned.zip")
env = BomberEnv()

# 测试
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### Q6: 如何调整模型参数

**BC参数调优**：
```python
# 在bc_imitation.py中修改
bc_trainer = bc.BC(
    ...
    policy_kwargs={
        "net_arch": [dict(pi=[512, 512], vf=[512, 512])],  # 增加网络容量
    }
)
```

**PPO参数调优**：
```python
# 在train_bc_ppo.py中修改
ppo_model = PPO(
    ...
    learning_rate=1e-4,  # 学习率
    gamma=0.99,          # 折扣因子
    gae_lambda=0.95,     # GAE参数
    clip_range=0.2,      # PPO裁剪范围
    ent_coef=0.01,       # 熵系数（鼓励探索）
)
```

---

## 📝 最佳实践

### 1. 数据收集

- ✅ 收集多样化的游戏场景
- ✅ 确保专家AI行为一致性
- ✅ 至少收集50-100局游戏
- ✅ 定期检查数据质量

### 2. BC训练

- ✅ 使用验证集监控过拟合
- ✅ 训练到准确率>80%
- ✅ 观察动作分布是否合理
- ✅ 保存最佳模型而非最后模型

### 3. PPO微调

- ✅ 从较小的学习率开始
- ✅ 使用多个并行环境加速训练
- ✅ 监控TensorBoard中的奖励曲线
- ✅ 定期评估模型性能
- ✅ 保存训练检查点

### 4. 超参数建议

**BC训练**：
- epochs: 50-200
- batch_size: 64-128
- learning_rate: 1e-3 (Adam优化器)

**PPO微调**：
- total_timesteps: 500k-2M
- learning_rate: 1e-4 ~ 3e-4
- n_steps: 2048-4096
- batch_size: 64-128
- n_epochs: 10
- n_envs: 4-8

---

## 🚀 快速开始示例

完整的训练命令示例：

```bash
# 1. 收集数据（假设已经修改好数据收集逻辑）
python collect_expert_data.py --episodes 100 --save_dir ./expert_data

# 2. 完整训练流程
python train_bc_ppo.py \
    --mode all \
    --expert_data ./expert_data \
    --bc_epochs 100 \
    --bc_batch_size 128 \
    --ppo_timesteps 1000000 \
    --ppo_lr 3e-4 \
    --n_envs 4 \
    --device auto \
    --evaluate \
    --n_eval_episodes 20

# 3. 查看训练日志
tensorboard --logdir ./logs/ppo
```

---

## 📚 参考资料

- [Stable Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [Imitation 库文档](https://imitation.readthedocs.io/)
- [Behavioral Cloning 论文](https://arxiv.org/abs/1011.0686)
- [PPO 论文](https://arxiv.org/abs/1707.06347)

---

## 🤝 贡献

如有问题或建议，欢迎提Issue或PR！

