import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件，指定列之间以 \t 分隔
# file_path = "../pseudo/01.20-21.29.05/gen_apigraph_cnt200_001_warm_lr0.003_sgd_step_0.95_e250_adam_wlr0.00015_we100_test_2013-01_2018-12_cnt200_lifetime.csv"  # 替换为你的文件路径
# file_path = "../cade/01.20-21.54.52/cade_apigraph_127_warm_lr0.0001_adam_cosine_1_e100_wlr5e-05_we50_mwlr5e-05_mwe50_test_2013-01_2018-12_cnt200_lifetime.csv"
# file_path = "../svm/01.20-21.54.55/gen_apigraph_hi-dist_svm_transcend_cred_cnt200_088_warm_lr0.003_sgd_step_0.95_e100_adam_wlr0.00015_we100_test_2013-01_2018-12_cnt200_lifetime.csv"
file_path = "../resnet/01.20-21.28.05/gen_apigraph_cnt200_002_warm_lr0.0009_e25_wlr0.000045_we25_test_2013-01_2018-12_cnt200_lifetime.csv"
df = pd.read_csv(file_path, sep="\t")

# 统计 'long' 列中每个值的出现行数
long_counts = df['long'].value_counts()

# 计算 'long' 的平均值
average_count = long_counts.mean()

# 计算总行数
total_rows = len(df)

# 绘制柱状图
plt.figure(figsize=(10, 6))
long_counts.sort_index().plot(kind="bar", color="lightblue", edgecolor="black")

# 添加水平虚线标注平均值
plt.axhline(average_count, color="gray", linestyle="--", linewidth=1, label=f"total_rows:{total_rows} Average: {average_count:.2f}")

# 添加总行数信息到标题
# plt.title(f"Counts of Each Long Value (Total Rows: {total_rows})", fontsize=14)

# 添加轴标签和图例
plt.xlabel("Long", fontsize=12)
plt.ylabel("Counts", fontsize=12)
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()
