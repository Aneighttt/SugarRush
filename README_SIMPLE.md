# 🎮 SugarRush 炸弹人AI训练系统

基于BC（行为克隆）+ PPO强化学习的炸弹人2v2 AI训练系统。

## 📋 快速开始

### 1. 收集专家数据（150局已完成✅）

```bash
# 启动游戏服务器（另一个终端）
cd demo/糖豆战阵（mac）
./start.sh

# 启动数据收集
python robot.py
```

数据会自动保存到 `./expert_data/expert_1/` 和 `./expert_data/expert_3/`

### 2. 训练BC模型

```bash
# 使用imitation库训练（推荐）
python train_bc_imitation.py --n_epochs 50 --batch_size 64

# 或使用自定义BC训练器
python bc_trainer.py
```

训练完成后，模型保存在 `./bc_models_imitation/`

### 3. PPO微调（可选）

```bash
python train_ppo.py --bc_model ./bc_models_imitation/bc_ppo_ready.zip
```

## 📂 项目结构

```
SugarRush/
├── robot.py                        # Flask服务器，接收游戏帧
├── realtime_expert_collector.py    # 实时专家数据收集器
├── bc_data_collector.py           # 数据收集工具类
├── train_bc_imitation.py          # BC训练（imitation库）
├── bc_trainer.py                  # BC训练（自定义）
├── train_ppo.py                   # PPO微调
├── environment.py                 # Gymnasium环境
├── config.py                      # 配置参数
├── data_models.py                 # 游戏数据结构
├── frame_processor.py             # 帧预处理
└── expert_data/                   # 专家数据目录
    ├── expert_1/                  # 专家1的数据
    └── expert_3/                  # 专家3的数据
```

## 🎯 核心组件

### 数据收集
- **实时收集**：从游戏服务器实时接收帧
- **多专家支持**：同时收集多个专家AI的数据
- **自动保存**：每局游戏结束自动保存

### 观察空间
- **grid_view**: (14, 16, 28) - 14通道地图信息
- **player_state**: (10,) - 玩家状态

### 动作空间
- **MultiDiscrete([5, 2, 5])**
  - 方向 (5): 不动/上/下/左/右
  - 炸弹 (2): 不放/放
  - 速度 (5): 相对速度档位

## 🚀 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_epochs` | 50 | BC训练轮数 |
| `--batch_size` | 64 | 批大小 |
| `--lr` | 0.001 | 学习率 |
| `--data_dir` | ./expert_data | 数据目录 |
| `--output_dir` | ./bc_models_imitation | 输出目录 |

## 📊 数据统计

- **收集局数**: 150局
- **专家数量**: 2个（expert_1, expert_3）
- **总样本数**: ~540,000 transitions
- **存储空间**: ~54 GB

## ⚙️ 配置说明

编辑 `config.py` 修改：
- 地图尺寸
- 观察空间维度
- 速度映射

## 🔧 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖：
- `gymnasium`
- `stable-baselines3`
- `imitation`
- `torch`
- `numpy`
- `flask`

## 📝 使用流程

1. **数据收集阶段**
   - 启动游戏服务器
   - 运行 `robot.py`
   - 等待收集足够数据（建议150-200局）

2. **BC训练阶段**
   - 运行 `train_bc_imitation.py`
   - 等待训练完成（约1-2小时，视硬件而定）
   - 获得BC模型

3. **PPO微调阶段**（可选）
   - 使用BC模型初始化PPO
   - 自我博弈继续训练
   - 超越专家水平

## 🐛 故障排除

### 问题：训练时显存不足
**解决**: 减小 `--batch_size` 到 32 或 16

### 问题：数据加载失败
**解决**: 检查 `expert_data/` 目录结构是否正确

### 问题：模型准确率低
**解决**: 
- 收集更多数据
- 增加训练轮数
- 调整学习率

## 📚 更多文档

- `TRAINING_GUIDE.md` - 详细训练指南
- `AUTO_COLLECTION_GUIDE.md` - 数据收集详解
- `QUICK_START.md` - 快速入门

## 📜 许可证

MIT License

---

**开始训练你的炸弹人AI吧！** 🎉

