# 🎯 BC模型用于PPO的完整指南

## ⚠️ 重要问题与解决方案

### 问题：BC训练使用的是Flattened观察

BC训练时使用了：
```python
env = FlattenObservation(BomberEnv())
# Observation: (6282,) = 14×16×28 + 10
```

但原始环境是：
```python
env = BomberEnv()
# Observation: Dict{"grid_view": (14,16,28), "player_state": (10,)}
```

### ✅ 解决方案

有两种方式让PPO使用BC初始化的模型：

---

## 方案1：PPO也使用Flattened环境（推荐）

### 优势
- ✅ 简单直接
- ✅ 与BC训练一致
- ✅ 可以直接加载bc_ppo_ready.zip

### 代码

```python
from stable_baselines3 import PPO
from environment import BomberEnv
from gymnasium.wrappers import FlattenObservation

# 创建flattened环境
base_env = BomberEnv()
env = FlattenObservation(base_env)

# 加载BC初始化的PPO模型
model = PPO.load("./bc_models_imitation/bc_ppo_ready.zip", env=env)

# 继续PPO训练
model.learn(total_timesteps=1_000_000)

# 保存
model.save("./ppo_models/ppo_finetuned.zip")
```

### 使用训练好的模型

```python
# 预测时也需要flatten
from gymnasium.wrappers import FlattenObservation

env = FlattenObservation(BomberEnv())
model = PPO.load("./ppo_models/ppo_finetuned.zip")

obs = env.reset()
action, _ = model.predict(obs)
```

---

## 方案2：创建支持Dict的PPO模型

### 优势
- ✅ 保持原始Dict observation space
- ✅ 更符合原始环境设计

### 劣势
- ⚠️ BC策略参数转换复杂
- ⚠️ 需要手动处理

### 代码

```python
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputPolicy
from environment import BomberEnv
import torch

# 1. 创建Dict环境的PPO
env = BomberEnv()  # 不用FlattenObservation

model = PPO(
    policy=MultiInputPolicy,  # 支持Dict observation
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1
)

# 2. 加载BC策略参数（需要手动映射）
bc_policy = torch.load("./bc_models_imitation/bc_policy.pt")

# 注意：这里需要手动处理flatten到Dict的映射
# 比较复杂，不推荐

# 3. 训练
model.learn(total_timesteps=1_000_000)
```

---

## 🎯 推荐流程

### Step 1: 确认BC模型已保存

```bash
ls -lh ./bc_models_imitation/
# 应该看到:
# bc_policy.pth
# bc_policy.pt
# bc_ppo_ready.zip  ← 这个用于PPO微调
```

### Step 2: PPO微调训练

```python
from stable_baselines3 import PPO
from environment import BomberEnv
from gymnasium.wrappers import FlattenObservation

# 使用flattened环境
env = FlattenObservation(BomberEnv())

# 加载BC初始化的模型
model = PPO.load("./bc_models_imitation/bc_ppo_ready.zip", env=env)

# PPO自我对弈微调
model.learn(
    total_timesteps=1_000_000,
    log_interval=10,
    tb_log_name="ppo_finetune"
)

# 保存
model.save("./ppo_models/ppo_final.zip")
```

### Step 3: 在游戏中使用

```python
# robot.py中使用
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from environment import BomberEnv
import numpy as np

# 加载模型
env = FlattenObservation(BomberEnv())
model = PPO.load("./ppo_models/ppo_final.zip")

# 在handle_command中
def handle_command():
    # ... 获取frame
    
    # 预处理
    obs = preprocess_observation_dict(frame)
    
    # Flatten观察
    flattened_obs = np.concatenate([
        obs['grid_view'].flatten(),
        obs['player_state']
    ])
    
    # 预测
    action, _ = model.predict(flattened_obs, deterministic=False)
    
    # action是[direction, bomb, speed]
    # ... 转换为游戏命令
```

---

## ⚡ 快速验证

运行这个脚本验证兼容性：

```python
# test_ppo.py
from stable_baselines3 import PPO
from environment import BomberEnv
from gymnasium.wrappers import FlattenObservation
import os

# 检查文件
if not os.path.exists("./bc_models_imitation/bc_ppo_ready.zip"):
    print("❌ bc_ppo_ready.zip不存在，请先完成BC训练")
    exit(1)

# 测试加载
try:
    env = FlattenObservation(BomberEnv())
    model = PPO.load("./bc_models_imitation/bc_ppo_ready.zip", env=env)
    print("✅ PPO模型加载成功！")
    
    # 测试预测
    obs = env.reset()
    action, _ = model.predict(obs)
    print(f"✅ 预测成功！动作: {action}")
    
    print("\n可以开始PPO微调了！")
    
except Exception as e:
    print(f"❌ 出错: {e}")
    import traceback
    traceback.print_exc()
```

---

## 📝 总结

### ✅ BC模型可以用于PPO，但需要注意

1. **环境一致性**: PPO训练和使用时都要用`FlattenObservation`
2. **文件确认**: 确保`bc_ppo_ready.zip`已生成
3. **观察处理**: 预测时需要flatten观察

### 推荐的完整流程

```
BC训练 (Flattened环境)
    ↓
保存 bc_ppo_ready.zip
    ↓
PPO微调 (Flattened环境)
    ↓
游戏使用 (手动flatten观察)
```

---

**关键点：PPO可以使用BC模型，但整个流程都要保持观察空间的一致性（都用Flattened）** ✅

