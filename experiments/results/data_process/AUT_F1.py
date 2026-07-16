import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 文件路径列表（替换为你的文件路径）
file_paths = [
    "../cade/01.20-21.54.52/cade_apigraph_127_warm_lr0.0001_adam_cosine_1_e100_wlr5e-05_we50_mwlr5e-05_mwe50_test_2013-01_2018-12_cnt200.csv",
    "../pseudo/01.20-21.29.05/gen_apigraph_cnt200_001_warm_lr0.003_sgd_step_0.95_e250_adam_wlr0.00015_we100_test_2013-01_2018-12_cnt200.csv",
    "../svm/01.20-21.54.55/gen_apigraph_hi-dist_svm_transcend_cred_cnt200_088_warm_lr0.003_sgd_step_0.95_e100_adam_wlr0.00015_we100_test_2013-01_2018-12_cnt200.csv",
    "../resnet/01.20-21.28.05/gen_apigraph_cnt200_002_warm_lr0.0009_e25_wlr0.000045_we25_test_2013-01_2018-12_cnt200.csv"
]
labels = ['Cade', 'Pseudo', 'Svm', 'Resnet']

# 用于存储每个文件的 AUT(F1) 数据
aut_data = []

# 读取每个文件并提取 AUT(F1) 列
for file_path in file_paths:
    df = pd.read_csv(file_path, sep='\t')
    if 'AUT(F1)' in df.columns:
        aut_data.append(df['AUT(F1)'])
    else:
        print(f"Warning: 'AUT(F1)' column not found in {file_path}")

# 绘制主图
plt.figure(figsize=(12, 8))

for i, data in enumerate(aut_data):
    plt.plot(data, label=f"{labels[i]}")

# 主图设置
plt.xlabel("Index", fontsize=12)
plt.ylabel("AUT(F1)", fontsize=12)
plt.legend()

# 添加嵌套图（右下角放大后半部分数据）
ax_inset = inset_axes(plt.gca(), width="50%", height="40%", loc="lower right")  # 嵌套图位置和大小

for i, data in enumerate(aut_data):
    # 放大后半部分数据
    start_idx = len(data) // 2
    ax_inset.plot(range(start_idx, len(data)), data[start_idx:], label=f"{labels[i]}")

# 嵌套图设置
# ax_inset.set_title("Zoomed-in View", fontsize=10)
ax_inset.tick_params(labelsize=8)
ax_inset.grid(alpha=0.5)

# 显示图表
plt.tight_layout()
plt.show()
