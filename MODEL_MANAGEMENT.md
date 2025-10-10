# 模型管理指南

## 📁 当前模型目录结构

```
SugarRush/
├── bc_models_imitation/        # 原版BC模型 (prob=0.48)
│   ├── bc_policy.pth           # PyTorch权重
│   ├── bc_policy.pt            # 完整对象
│   └── bc_ppo_ready.zip        # PPO格式 ← robot.py会用
│
├── bc_models_weighted/         # 加权BC模型 (即将训练, 预期prob=0.58-0.65)
│   ├── bc_policy_best.pth      # 最佳权重
│   ├── bc_policy_final.pth     # 最终权重
│   └── bc_ppo_weighted.zip     # PPO格式 ← robot.py会优先用
│
└── ppo_models/                 # PPO微调模型 (未来)
    └── ppo_finetuned.zip       # ← robot.py最优先
```

---

## 🔄 模型自动选择逻辑

`robot.py` 会按以下优先级自动加载模型：

1. **PPO微调模型** (最优先)
   - 路径: `./ppo_models/ppo_finetuned.zip`
   - 状态: 暂无 (需要PPO fine-tuning)

2. **加权BC模型** (次优先) ⭐
   - 路径: `./bc_models_weighted/bc_ppo_weighted.zip`
   - 状态: 即将训练
   - 特点: 提升炸弹&减速权重

3. **原版BC模型** (备选)
   - 路径: `./bc_models_imitation/bc_ppo_ready.zip`
   - 状态: 已训练完成
   - 准确率: prob=0.48

4. **随机AI** (兜底)
   - 如果所有模型都不存在，使用随机策略

---

## ✅ 不需要删除旧模型

### 原因：
1. **自动选择** - `robot.py` 会自动用最好的模型
2. **效果对比** - 保留旧模型可以A/B测试
3. **回退选项** - 如果新模型有问题可以快速回退
4. **存储占用小** - 每个模型约10-50MB

### 训练新模型后的行为：
```bash
./TRAIN_WEIGHTED.sh  # 训练加权模型
# ↓
# 新模型保存到 bc_models_weighted/bc_ppo_weighted.zip
# ↓
# robot.py 自动检测并优先使用新模型 ✅
# ↓
# 旧模型 (bc_models_imitation/) 仍然保留作为备份
```

---

## 🎛️ 手动切换模型

### 方法1: 重命名文件（临时禁用某个模型）
```bash
# 禁用加权模型，回退到原版模型
mv bc_models_weighted/bc_ppo_weighted.zip bc_models_weighted/bc_ppo_weighted.zip.bak

# 恢复
mv bc_models_weighted/bc_ppo_weighted.zip.bak bc_models_weighted/bc_ppo_weighted.zip
```

### 方法2: 修改 robot.py 中的优先级
编辑 `robot.py` 第46-56行，调整 `if-elif` 顺序

---

## 🗑️ 如果确实要删除（节省空间）

### 安全删除策略（从旧到新）：

#### 1️⃣ 可删除：中间训练产物
```bash
# 删除非PPO格式的权重文件（只保留 .zip）
rm bc_models_imitation/bc_policy.pth
rm bc_models_imitation/bc_policy.pt
rm bc_models_weighted/bc_policy_final.pth  # 保留 best 版本

# 节省空间: ~20-40MB
```

#### 2️⃣ 谨慎删除：原版BC模型（如果加权模型效果好）
```bash
# 等加权模型测试通过后再删
rm -rf bc_models_imitation/

# 节省空间: ~50-100MB
```

#### 3️⃣ 不要删除：
- ❌ `bc_models_weighted/` - 最新最好的模型
- ❌ `expert_data/` - 专家数据（可能还需要重新训练）
- ❌ `ppo_models/` - PPO微调模型（如果存在）

---

## 📊 存储空间估算

```
expert_data/           ~800MB-1.5GB  (300 episodes)
bc_models_imitation/   ~50-100MB     (原版BC)
bc_models_weighted/    ~50-100MB     (加权BC)
ppo_models/            ~50-100MB     (PPO微调, 未来)
─────────────────────────────────────────
总计:                  ~1-2GB
```

### 如果存储紧张：
1. 删除 `expert_data/expert_*/expert_*_stats.json` (仅统计文件) - 节省 ~10MB
2. 压缩 `expert_data/` 目录后备份到云端，删除本地副本 - 节省 ~1GB
3. 只保留最新的BC模型，删除旧版本 - 节省 ~50MB

---

## 🔍 查看当前使用的模型

启动 `robot.py` 时会显示：
```
✅ 加权BC模型 (提升炸弹&减速权重)加载成功: ./bc_models_weighted/bc_ppo_weighted.zip
   观察空间: Flattened((6282,))
   动作空间: MultiDiscrete([5 2 5])
```

---

## 🎯 推荐工作流

### 当前阶段：
```
1. 保留原版BC模型 (bc_models_imitation/)
   ↓
2. 训练加权BC模型 (bc_models_weighted/)
   ↓
3. 测试对比两个模型的游戏表现
   ↓
4. 确认加权模型更好后，可删除原版BC
```

### 未来阶段（PPO微调）：
```
1. 使用加权BC模型初始化PPO
   ↓
2. PPO训练完成后保存到 ppo_models/
   ↓
3. robot.py 自动切换到PPO模型
   ↓
4. 可以删除所有BC模型（但建议保留1个作为备份）
```

---

## 📝 总结

### ✅ 推荐做法：
- **保留所有模型** - 方便对比和回退
- **让 robot.py 自动选择** - 无需手动干预
- **定期清理中间产物** - 删除 .pth 保留 .zip

### ❌ 不推荐：
- 在训练前删除旧模型
- 删除专家数据（除非已确认不需要重新训练）
- 手动修改模型加载逻辑（除非有特殊需求）

---

## 🚀 下一步

现在可以安心训练新模型：
```bash
./TRAIN_WEIGHTED.sh
```

训练完成后，`robot.py` 会自动使用新模型，无需任何手动操作！

