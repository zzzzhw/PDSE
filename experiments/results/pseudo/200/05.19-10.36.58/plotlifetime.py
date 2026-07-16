# 读取lifetime.csv文件
import pandas as pd
# 设置当前目录为工作目录
import os
# 根据long列的值分组绘制柱状图
import matplotlib.pyplot as plt
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 读取lifetime.csv文件
df = pd.read_csv('gen_apigraph_cnt200_001_warm_lr0.003_sgd_step_0.95_e250_adam_wlr0.00015_we100_test_2013-01_2018-12_cnt200_lifetime.csv', sep='\t')

# 根据long列的值计算每个long值有多少行
long_counts = df['long'].value_counts()

# 计算加权平均值
total_samples = sum(long_counts.values)
weighted_sum = sum(period * count for period, count in long_counts.items())
mean_value = weighted_sum / total_samples if total_samples > 0 else 0

# 绘制柱状图
plt.figure(figsize=(10, 6))
bars = plt.bar(long_counts.index, long_counts.values, color='skyblue')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')

# 添加均值线
plt.axhline(y=mean_value, color='red', linestyle='--', label=f'survival periods_mean: {mean_value:.2f}')

# 设置图表标题和标签
plt.xlabel('survival_period', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(long_counts.index, rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 添加图例
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()

