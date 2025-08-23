# -*- coding: utf-8 -*-
"""
此脚本用于分析三种细胞系（A549, 293T, THP-1）在感染嗜肺军团菌后的多项指标数据。
功能包括：
1. 数据加载与预处理 (已增加NaN值处理)。
2. 主成分分析（PCA）以可视化不同细胞系的总体反应模式差异。
3. 在PCA图上为每个分组绘制95%置信椭圆。
4. 使用PERMANOVA检验总体及两两分组间的统计显著性，并进行多重检验校正。
5. 对每个独立指标进行重复测量方差分析（RM ANOVA）和配对t检验。
6. 为每个指标生成箱线图并标注统计结果。

使用前请确保已安装所需库:
pip install pandas pingouin matplotlib seaborn scikit-learn scikit-bio
"""

import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skbio.stats.distance import permanova, DistanceMatrix
from sklearn.metrics import pairwise_distances
import numpy as np
import itertools
from matplotlib.patches import Ellipse
import matplotlib.transforms as mtransforms

# --- 0. 环境设置 ---

# 定义输出目录用于保存图表
output_dir = Path('analysis_plots_corrected')
output_dir.mkdir(exist_ok=True)

# 设置matplotlib支持中文显示，并确保字体嵌入PDF
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一个常用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
plt.rcParams['pdf.fonttype'] = 42          # 确保PDF中的文字可编辑，而不是转换为轮廓

def confidence_ellipse(x, y, ax, n_std=1.96, facecolor='none', **kwargs):
    """
    根据给定的x, y数据点创建并绘制一个置信椭圆。
    n_std=1.96 对应 95% 置信区间。
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = mtransforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# --- 1. 数据加载与准备 ---

# 加载数据集
try:
    df = pd.read_csv('PCA分析数据集.csv', encoding='gbk')
except FileNotFoundError:
    print("错误: 'PCA分析数据集.csv' 未找到。请确保该文件与脚本在同一目录下。")
    exit()

# 重命名列
df.columns = [
    'SampleID', 'Internalization_Rate', 'Intracellular_bacteria_fold_change_12h',
    'Intracellular_bacteria_fold_change_24h', 'Intracellular_bacteria_fold_change_48h',
    'Live_percent_6h', 'Live_percent_12h', 'Live_percent_24h', 'Live_percent_48h',
    'IL-1a_6h', 'IL-1b_6h', 'IL-6_6h', 'IL-8_6h', 'IL-10_6h', 'TNFa_6h',
    'IL-1a_12h', 'IL-1b_12h', 'IL-6_12h', 'IL-8_12h', 'IL-10_12h', 'TNFa_12h',
    'IL-1a_24h', 'IL-1b_24h', 'IL-6_24h', 'IL-8_24h', 'IL-10_24h', 'TNFa_24h',
    'IL-1a_48h', 'IL-1b_48h', 'IL-6_48h', 'IL-8_48h', 'IL-10_48h', 'TNFa_48h'
]

# 提取分组信息
df['StrainID'] = df['SampleID'].apply(lambda x: x.split('_')[1])
df['CellType'] = df['SampleID'].apply(lambda x: x.split('_')[0])

# 准备数值型数据
features = df.columns.drop(['SampleID', 'StrainID', 'CellType'])
X = df[features].copy()

# 检查并填充NaN值
if X.isnull().values.any():
    print("警告: 数据中检测到缺失值(NaN)。将使用每列的平均值进行填充。")
    X = X.fillna(X.mean())

# --- 2. PCA分析与统计检验 ---

print("--- 正在执行PCA分析 ---")

# 2.1 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2.2 执行PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# 2.3 创建PCA结果的DataFrame
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['CellType'] = df['CellType']
pca_df['StrainID'] = df['StrainID']

explained_variance = pca.explained_variance_ratio_
print(f"PC1 解释的方差: {explained_variance[0]:.2%}")
print(f"PC2 解释的方差: {explained_variance[1]:.2%}")

# 2.4 PERMANOVA 统计分析
numpy_distance_matrix = pairwise_distances(X_scaled, metric='euclidean')
numpy_distance_matrix = (numpy_distance_matrix + numpy_distance_matrix.T) / 2.0
skbio_distance_matrix = DistanceMatrix(numpy_distance_matrix, ids=df.index)

# 2.4.1 总体 PERMANOVA
print("\n--- 总体 PERMANOVA 统计结果 ---")
permanova_results_overall = permanova(skbio_distance_matrix, df['CellType'])
permanova_p_value_overall = permanova_results_overall['p-value']
print(permanova_results_overall)

# 2.4.2 两两比较 PERMANOVA
print("\n--- 两两比较 PERMANOVA 统计结果 ---")
cell_types = df['CellType'].unique()
pairs = list(itertools.combinations(cell_types, 2))
pairwise_p_values = []

for group1, group2 in pairs:
    # 筛选出当前比较对的数据
    pair_indices = df[df['CellType'].isin([group1, group2])].index
    pair_grouping = df.loc[pair_indices, 'CellType']
    
    # 从总距离矩阵中提取子矩阵
    pair_dist_matrix = skbio_distance_matrix.filter(pair_indices)
    
    # 执行PERMANOVA
    result = permanova(pair_dist_matrix, pair_grouping)
    pairwise_p_values.append(result['p-value'])
    print(f"{group1} vs {group2}: p-value = {result['p-value']:.4f}")

# 对p值进行多重检验校正 (Bonferroni)
p_corrected = pg.multicomp(pairwise_p_values, method='bonf')[1]
pairwise_results_corrected = list(zip(pairs, p_corrected))
print("\n--- 校正后的两两比较p值 (Bonferroni) ---")
for (g1, g2), p_val in pairwise_results_corrected:
    print(f"{g1} vs {g2}: p-corr = {p_val:.4f}")


# --- 3. PCA 可视化 (带置信椭圆和统计标注) ---

print("\n--- 正在生成并保存PCA图表 ---")
fig, ax = plt.subplots(figsize=(13, 11))

# 绘制散点图
sns.scatterplot(
    x='PC1', y='PC2', hue='CellType', style='CellType',
    data=pca_df, s=150, alpha=0.8, palette='viridis', ax=ax
)

# 为每个组添加95%置信椭圆
colors = sns.color_palette('viridis', n_colors=len(cell_types))
color_map = dict(zip(cell_types, colors))

for cell_type in cell_types:
    group_data = pca_df[pca_df['CellType'] == cell_type]
    if len(group_data) >= 3:
        confidence_ellipse(group_data['PC1'], group_data['PC2'], ax,
                           edgecolor=color_map[cell_type], linewidth=2)
    else:
        print(f"警告: 细胞系 {cell_type} 的数据点少于3个，无法绘制置信椭圆。")

ax.set_title('不同细胞系感染反应模式的PCA分析', fontsize=18)
ax.set_xlabel(f'主成分 1 ({explained_variance[0]:.2%})', fontsize=14)
ax.set_ylabel(f'主成分 2 ({explained_variance[1]:.2%})', fontsize=14)
ax.legend(title='细胞系', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)

# 在图上标注统计结果
stats_text_overall = f"Overall PERMANOVA: p = {permanova_p_value_overall:.4f}"
stats_text_pairwise = "Pairwise PERMANOVA (p-corr):\n"
for (g1, g2), p_val in pairwise_results_corrected:
    stats_text_pairwise += f"  {g1} vs {g2}: {p_val:.4f}\n"

plt.text(0.02, 0.02, stats_text_overall, transform=ax.transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment='left',
         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

plt.text(0.98, 0.02, stats_text_pairwise, transform=ax.transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.7))

# 保存PCA图
pca_plot_filename = output_dir / 'PCA_with_Ellipse_and_Stats.pdf'
plt.savefig(pca_plot_filename, format='pdf', bbox_inches='tight')
plt.close()

print(f"PCA图表已保存至: {pca_plot_filename}")


# --- 4. 各变量的统计分析 (重复测量ANOVA和配对T检验) ---

results = {}
print("\n--- 正在对各变量进行统计分析 ---")
df_no_nan = df.drop(columns=features).join(X)

for var in features:
    aov_result = pg.rm_anova(data=df_no_nan, dv=var, within='CellType', subject='StrainID', detailed=True)
    paired_tests_result = pg.pairwise_tests(data=df_no_nan, dv=var, within='CellType', subject='StrainID', padjust='fdr_bh')
    results[var] = {'rm_anova': aov_result, 'paired_tests': paired_tests_result}
    
    print(f"\n--- {var} 的分析结果 ---")
    print("\n重复测量ANOVA:")
    print(aov_result[aov_result['Source'] == 'CellType'])
    print("\n配对 T-检验 (FDR校正):")
    print(paired_tests_result[['A', 'B', 'p-corr', 'BF10']])

# --- 5. 各变量的可视化 (箱线图) ---

print("\n--- 正在生成并保存各变量的箱线图 ---")
for var in features:
    plt.figure(figsize=(7, 7))
    sns.boxplot(data=df_no_nan, x='CellType', y=var, order=['A549', '293T', 'THP-1'], palette='pastel')
    sns.stripplot(data=df_no_nan, x='CellType', y=var, order=['THP-1', 'A549', '293T'], color='black', jitter=0.2, alpha=0.6,s=10, marker='o')
    
    plt.title(f'{var} 在不同细胞系中的比较', fontsize=16)
    plt.ylabel(var, fontsize=12)
    plt.xlabel('细胞系', fontsize=12)
    
    stats_text = "配对T检验 p值 (FDR校正):\n"
    test_results_for_var = results[var]['paired_tests']
    for _, row in test_results_for_var.iterrows():
        p_val = row['p-corr']
        sig_symbol = 'ns'
        if p_val < 0.001: sig_symbol = '***'
        elif p_val < 0.01: sig_symbol = '**'
        elif p_val < 0.05: sig_symbol = '*'
        stats_text += f"{row['A']} vs {row['B']}: p={p_val:.4f} ({sig_symbol})\n"
        
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.7))

    plt.tight_layout()
    plot_filename = output_dir / f'{var}_boxplot.pdf'
    plt.savefig(plot_filename, format='pdf')
    plt.close()

print(f"\n分析完成。所有图表已保存至 '{output_dir}' 文件夹。")