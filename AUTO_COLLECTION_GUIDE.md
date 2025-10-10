# 🤖 自动收集模式使用指南

## ✨ 新功能：自动保存

现在系统支持**自动检测游戏结束并保存数据**！

### 特性
- ✅ 自动检测tick达到1800（或接近）
- ✅ 自动检测新一局开始（tick重置）
- ✅ 自动保存每一局数据
- ✅ 实时显示已保存的Episode数
- ✅ 游戏服务器连续运行，无需手动干预

## 🚀 使用方法

### 一键启动

```bash
python robot.py
```

**就这么简单！** 让游戏服务器一直运行，系统会自动收集所有数据。

## 📊 运行输出示例

### 启动时

```
--- Prediction agent model loaded successfully. ---

✅ 实时专家数据收集器已初始化
   自动保存: 启用
   游戏最大tick: 1800

--- 专家数据收集已启用（自动模式）---
--- 游戏结束时会自动保存数据 ---
```

### 游戏进行中

```
🎮 新游戏开始 (Match ID: abc123, Episode: 1)

识别到2个专家AI:
  专家 player_A (Team: 2)
  专家 player_B (Team: 2)

📈 进度 - Episode 1, Frame 100, Tick 100, Transitions 198
📈 进度 - Episode 1, Frame 200, Tick 200, Transitions 398
📈 进度 - Episode 1, Frame 300, Tick 300, Transitions 598
...
📈 进度 - Episode 1, Frame 1700, Tick 1700, Transitions 3398
```

### 自动保存（游戏结束）

```
🔄 检测到Tick重置 (1799 -> 5)，保存上一局数据...

============================================================
💾 保存Episode 1
   总帧数: 1799
   最后Tick: 1799

✅ 数据已保存到: expert_data/expert_player_A/expert_player_A_ep0000.pkl
   总episode数: 1
   总transition数: 1799
   
   动作分布:
     方向-上: 234 (13.0%)
     方向-下: 189 (10.5%)
     方向-左: 421 (23.4%)
     方向-右: 655 (36.4%)
     方向-不动: 300 (16.7%)
     不放炸弹: 1623 (90.2%)
     放炸弹: 176 (9.8%)
     速度-最大速度: 1200 (66.7%)
     速度-慢速: 250 (13.9%)
     速度-中速: 349 (19.4%)

✅ Episode 1 所有专家数据已保存
   📁 expert_data/expert_player_A/expert_player_A_ep0000.pkl
   📁 expert_data/expert_player_B/expert_player_B_ep0000.pkl

📊 总进度：已保存 1 个Episode
============================================================

🎮 新游戏开始 (Episode: 2)
```

### 继续收集

```
📈 进度 - Episode 2, Frame 100, Tick 100, Transitions 198
...

🔄 检测到Tick重置 (1799 -> 5)，保存上一局数据...

============================================================
💾 保存Episode 2
...
📊 总进度：已保存 2 个Episode
============================================================

🎮 新游戏开始 (Episode: 3)
```

## 📁 文件命名

文件会自动编号（4位数字）：

```
expert_data/
├── expert_player_A/
│   ├── expert_player_A_ep0000.pkl
│   ├── expert_player_A_ep0001.pkl
│   ├── expert_player_A_ep0002.pkl
│   └── ...
└── expert_player_B/
    ├── expert_player_B_ep0000.pkl
    ├── expert_player_B_ep0001.pkl
    └── ...
```

最多支持9999个Episode！

## 🎯 自动检测机制

系统通过两种方式检测游戏结束：

### 方式1：Match ID变化
```python
if current_match_id != previous_match_id:
    # 新一局开始，保存上一局
    save_episode()
```

### 方式2：Tick重置
```python
if current_tick < last_tick - 100:
    # Tick突然下降，说明新一局开始
    save_episode()
```

### 方式3：Tick接近最大值
```python
if current_tick >= 1800 - 10:
    # 游戏即将结束，提醒
    print("⏰ 游戏即将结束...")
```

## ⚙️ 配置选项

如果需要修改参数，编辑`robot.py`:

```python
expert_collector = enable_data_collection(
    save_dir="./expert_data",  # 数据保存路径
    auto_save=True,             # 是否自动保存
    max_ticks=1800              # 游戏最大tick（根据实际游戏调整）
)
```

### 修改监控频率

编辑`realtime_expert_collector.py`:

```python
def __init__(self, save_dir: str = "./expert_data", 
             save_interval: int = 100,  # 每100帧打印一次进度
             ...):
```

改为50，每50帧打印一次：
```python
save_interval: int = 50,
```

## 📊 监控收集进度

### 方法1：查看输出

直接看robot.py的输出，会显示：
- 当前Episode
- 当前Frame
- 当前Tick
- 已收集的Transitions数

### 方法2：实时监控文件

另开一个终端：

```bash
# 监控数据大小
watch -n 5 'du -sh expert_data/'

# 查看文件数量
watch -n 10 'ls -l expert_data/*/ | grep -c pkl'
```

### 方法3：查看统计文件

```bash
# 查看最新统计
cat expert_data/expert_*/expert_*_stats.json | jq
```

## 🛑 停止收集

### 方法1：等待目标完成

让程序一直运行，直到收集够数据（例如200局）。

### 方法2：手动停止

按 `Ctrl+C`，程序会：
1. 检查是否有未保存的数据
2. 如果有，自动保存
3. 打印最终统计
4. 退出

输出示例：
```
^C
============================================================
程序退出：检查未保存的数据
============================================================
⚠️  发现未保存的数据 (856 帧)

💾 保存Episode 101
...
📊 总进度：已保存 101 个Episode
============================================================

✅ 所有数据已保存
============================================================
实时专家数据收集统计
总Episodes: 101
...
```

## 💡 最佳实践

### 长时间收集

```bash
# 使用nohup在后台运行
nohup python robot.py > collection.log 2>&1 &

# 查看实时日志
tail -f collection.log

# 停止
pkill -f robot.py
```

### 分批收集

```bash
# 第1批：收集50局
python robot.py  # 运行到50局后Ctrl+C

# 第2批：继续收集
python robot.py  # 会从Episode 50开始编号
```

### 数据备份

```bash
# 定期备份
cp -r expert_data/ expert_data_backup_$(date +%Y%m%d)/

# 或使用rsync
rsync -av expert_data/ /path/to/backup/
```

## 🎯 收集目标建议

### 快速测试（1-2小时）
- 目标：10-20局
- 数据量：~440MB - 880MB
- 用途：验证系统正常

### 基础训练（4-8小时）
- 目标：50-100局
- 数据量：~2.2GB - 4.4GB
- 用途：初步BC训练

### 充分训练（12-20小时）
- 目标：200-300局
- 数据量：~8.6GB - 13GB
- 用途：高质量BC + PPO

### 最佳效果（40+小时）
- 目标：500+局
- 数据量：~22GB+
- 用途：专业级模型

## ✅ 优势

对比手动模式：

| 特性 | 手动模式 | 自动模式 |
|------|----------|----------|
| 需要人工干预 | ✅ 每局都要 | ❌ 无需 |
| 可能遗漏数据 | ✅ 容易 | ❌ 不会 |
| 连续收集 | ❌ 困难 | ✅ 简单 |
| 过夜收集 | ❌ 不行 | ✅ 可以 |
| 误操作风险 | ✅ 有 | ❌ 无 |

## 🔧 故障排查

### Q: 没有自动保存？

检查：
1. 是否有"自动保存: 启用"的输出？
2. Tick是否真的在变化？
3. Match ID是否变化？

调试：
```python
# 在process_frame中添加
print(f"Debug: tick={frame.current_tick}, last_tick={self.last_tick}")
```

### Q: 保存太频繁？

可能是tick异常波动，调整阈值：
```python
elif current_tick < self.last_tick - 100:  # 改为200
```

### Q: 文件名重复？

不会！使用4位数字编号（0000-9999），按episode_count递增。

## 🎓 总结

**自动模式下，你只需要**：
1. 启动robot.py
2. 让游戏服务器运行
3. 等待收集完成
4. Ctrl+C停止

**系统会自动**：
- 检测游戏结束
- 保存数据
- 开始新一局
- 打印进度

**完全解放双手，专注于训练和优化！** 🚀

