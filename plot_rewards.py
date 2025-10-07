import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib

# --- Matplotlib Configuration for Chinese Characters ---
# Use a font that supports Chinese characters. 'SimHei' is a common choice.
# You might need to change this to a font available on your system.
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# To display negative signs correctly
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_reward_distribution(log_dir="reward"):
    """
    Reads the reward log file from the specified directory and plots:
    1. The distribution of total rewards.
    2. A summary of total contributions from each reward component.
    Saves the plots in the same directory.
    """
    log_file = os.path.join(log_dir, "reward_log.csv")
    
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"错误: 未找到日志文件 '{log_file}'。")
        print("请先运行训练以生成日志文件。")
        return

    if 'total_reward' not in df.columns:
        print(f"错误: 在 '{log_file}' 中未找到 'total_reward' 列。")
        return

    # Set plot style
    sns.set_theme(style="whitegrid")

    # --- 图 1: 每步总奖励分布 ---
    plt.figure(figsize=(12, 6))
    ax = sns.histplot(df['total_reward'], bins=50, kde=False)
    plt.title('每步总奖励分布')
    plt.xlabel('总奖励')
    plt.ylabel('频率')
    plt.grid(True)

    # 在每个条形图上方添加数值标签
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 5),
                        textcoords='offset points')

    dist_path = os.path.join(log_dir, 'reward_distribution.png')
    plt.savefig(dist_path)
    print(f"奖励分布图已保存至 '{dist_path}'")

    # --- 图 2: 各奖励项总贡献 ---
    # 获取所有奖励项的列名
    reward_cols = [col for col in df.columns if col not in ['step', 'total_reward', 'reward_moving_avg']]
    # 计算每个奖励项的总和
    reward_sums = df[reward_cols].sum().sort_values()

    plt.figure(figsize=(12, 8))
    bars = reward_sums.plot(kind='barh', color=sns.color_palette("coolwarm", len(reward_sums)))
    plt.title('各奖励项总贡献')
    plt.xlabel('奖励总和')
    plt.ylabel('奖励项')
    plt.grid(axis='x')
    
    # 在条形图上添加数值标签
    for bar in bars.patches:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                 f' {bar.get_width():.2f}',
                 va='center')

    plt.tight_layout()
    summary_path = os.path.join(log_dir, 'reward_components_summary.png')
    plt.savefig(summary_path)
    print(f"奖励项贡献总览图已保存至 '{summary_path}'")

    plt.show()

if __name__ == '__main__':
    plot_reward_distribution()
