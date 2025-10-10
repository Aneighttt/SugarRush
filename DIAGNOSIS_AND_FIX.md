# BC模型效果不佳 - 诊断与解决方案

## 🔍 当前状态

### 模型信息
- ✅ 网络结构: `[256, 128, 64]` - 正确
- ✅ 参数更新: 99.99%参数非零 - 正常训练
- ✅ 观察处理: Flatten正确
- ❓ **训练效果未知** - 没有训练日志

### 症状
**模型表现和随机AI没有明显区别**

---

## 💡 可能的原因

### 1. prob_true_act太低 (最可能) ⭐⭐⭐⭐⭐

如果训练的`prob_true_act < 0.60`，模型基本无法正常工作。

**诊断方法**：
```bash
# 查找训练输出
find . -name "*.out" -o -name "*.log" | xargs grep -l "prob_true_act" 2>/dev/null
```

**解决方案**：
- 增加训练轮数到500-1000 epochs
- 降低学习率到0.0001
- 增大网络到`[512, 256, 128]`

---

### 2. 网络容量仍然不足 ⭐⭐⭐⭐

虽然已经从`[64, 64]`升级到`[256, 128, 64]`，但对于6282维输入可能还不够。

**对比**：
```
输入: 6282维
当前: [256, 128, 64]  - 第一层只有256维 (4%信息保留)
推荐: [512, 256, 128]  - 第一层512维 (8%信息保留)
理想: [1024, 512, 256] - 第一层1024维 (16%信息保留)
```

**解决方案**：
```python
# 修改 train_bc_weighted.py 第319行
policy_kwargs = dict(
    net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])
)
```

---

### 3. 训练轮数不够 ⭐⭐⭐

当前可能只训练了100-300 epochs，对于大网络来说不够。

**推荐轮数**：
- `[256, 128, 64]`: 500-1000 epochs
- `[512, 256, 128]`: 1000-2000 epochs

---

### 4. 学习率设置 ⭐⭐⭐

当前学习率`0.0003`可能对大网络来说过高或过低。

**推荐设置**：
- 初始300 epochs: `lr=0.0003`
- 后续精调: `lr=0.0001` 或 `0.00005`

---

### 5. 专家数据质量 ⭐⭐

如果专家AI本身不够强，收集的数据质量就不高。

**检查方法**：
```python
# 检查专家数据中的胜率
import pickle
data = pickle.load(open('expert_data/expert_1/expert_1_ep0001.pkl', 'rb'))
# 看episode是否能完成目标
```

---

## 🚀 立即改进方案

### 方案A: 增大网络 + 长时间训练 (推荐) ⭐⭐⭐⭐⭐

```bash
# 1. 修改网络结构
# 编辑 train_bc_weighted.py 第319行
policy_kwargs = dict(
    net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])
)

# 2. 删除旧模型
rm -rf bc_models_weighted/

# 3. 长时间训练
python train_bc_weighted.py \
    --n_epochs 1000 \
    --batch_size 128 \
    --lr 0.0002
```

**预期**：
- 训练时间: 8-12小时
- prob_true_act: 0.70-0.80+
- 显著的游戏效果提升

---

### 方案B: 继续训练当前模型 ⭐⭐⭐⭐

```bash
# 不修改网络，直接继续训练
python continue_bc_training.py \
    --checkpoint bc_models_weighted/bc_policy_weighted.pt \
    --epochs 500 \
    --batch_size 128 \
    --lr 0.0001
```

**预期**：
- 训练时间: 6-8小时
- prob_true_act: 可能提升到0.65-0.70
- 中等提升

---

### 方案C: 收集更多数据 ⭐⭐⭐

```bash
# 如果当前只有300局数据，收集到1000局
# 然后重新训练
```

---

## 📊 训练监控

### 关键指标

训练时需要关注：

```
| epoch | prob_true_act | direction_acc | bomb_acc | speed_acc |
|-------|---------------|---------------|----------|-----------|
| 100   | 0.48          | 0.70          | 0.30     | 0.50      | ❌ 太低
| 300   | 0.60          | 0.78          | 0.55     | 0.65      | ⚠️  勉强
| 500   | 0.70          | 0.85          | 0.70     | 0.75      | ✅ 可用
| 1000  | 0.78+         | 0.90+         | 0.80+    | 0.85+     | ✅ 优秀
```

**判断标准**：
- `prob_true_act < 0.55`: 模型基本无效
- `prob_true_act 0.55-0.65`: 勉强可用
- `prob_true_act 0.65-0.75`: 良好
- `prob_true_act > 0.75`: 优秀

---

## 🛠️ 快速修复脚本

### 1. 创建大网络训练脚本

```bash
cat > train_large_network.sh << 'EOF'
#!/bin/zsh

echo "================================"
echo "大网络BC训练 [512, 256, 128]"
echo "================================"

# 修改训练脚本
cp train_bc_weighted.py train_bc_large.py

# 用sed替换网络结构
sed -i '' 's/\[256, 128, 64\]/[512, 256, 128]/g' train_bc_large.py

# 训练
source ~/Desktop/workspace/venv/bin/activate

python train_bc_large.py \
    --n_epochs 1000 \
    --batch_size 128 \
    --lr 0.0002 \
    --output_dir ./bc_models_large

echo "✅ 训练完成"
EOF

chmod +x train_large_network.sh
./train_large_network.sh
```

---

### 2. 查看实际训练效果

```bash
# 查找所有可能的日志
find . -type f \( -name "*.out" -o -name "*.log" \) -exec grep -l "prob_true_act" {} \;

# 查看最近的训练记录
ls -lt | head -20
```

---

## 🎯 推荐行动

### 立即执行 (选择一个)：

#### 选项1: 大网络重新训练 (推荐，如果时间充裕)
```bash
# 1. 修改 train_bc_weighted.py
# 第319行改为: net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])

# 2. 删除旧模型
rm -rf bc_models_weighted/

# 3. 开始训练
python train_bc_weighted.py --n_epochs 1000 --batch_size 128 --lr 0.0002
```

#### 选项2: 继续训练当前模型 (推荐，如果想快速改进)
```bash
python continue_bc_training.py \
    --checkpoint bc_models_weighted/bc_policy_weighted.pt \
    --epochs 500 \
    --batch_size 128 \
    --lr 0.0001
```

---

## 🔬 验证改进

训练完成后，检查：

1. **查看训练日志**:
   ```bash
   tail -100 nohup.out | grep "prob_true_act"
   ```

2. **检查模型**:
   ```bash
   python inspect_model.py --model bc_models_weighted/bc_ppo_weighted.zip
   ```

3. **实际游戏测试**:
   ```bash
   python robot.py
   # 观察AI是否会：
   # - 主动放炸弹 💣
   # - 躲避危险 🏃
   # - 有目的性移动 🎯
   ```

---

## ❓ 仍然无效？

如果尝试以上方案后仍然无效，可能需要：

1. **检查frame_processor归一化**
   - 确认grid_view值域在[0, 1]
   - 确认player_state归一化正确

2. **检查专家数据**
   - 专家AI是否够强
   - 是否有足够多样性

3. **考虑更换方法**
   - 从PPO从头训练（无BC初始化）
   - 使用DQN/Rainbow等其他算法
   - 收集人类演示数据

---

**现在选择一个方案开始改进！** 🚀

推荐: **方案B (继续训练500 epochs)**，因为最快能看到效果。

