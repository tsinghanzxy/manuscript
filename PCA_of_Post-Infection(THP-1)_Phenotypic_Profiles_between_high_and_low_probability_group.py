# -*- coding: utf-8 -*-
"""
此脚本用于分析THP-1细胞系在感染嗜肺军团菌后的多项指标数据，并分为
Low_probability和High_probability两组进行比较。
功能包括：
1. 数据加载与预处理 (已增加NaN值处理)。
2. 主成分分析（PCA）以可视化不同分组的总体反应模式差异。
3. 在PCA图上为每个分组绘制95%置信椭圆。
4. 使用PERMANOVA检验总体及两两分组间的统计显著性。
5. 对每个独立指标或每个细胞因子-时间点组合进行统计分析。
6. 为每个指标或每个细胞因子生成箱线图，并标注统计结果。
7. 新增：将6个细胞因子按时间点组合，分别生成箱线图并进行统计比较。
8. 新增：为每个单独的细胞因子生成跨时间点的箱线图及统计分析。

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
import re

# --- 0. 环境设置 ---

# 定义输出目录用于保存图表
# 请注意：如果您希望输出到主目录或其他位置，请修改此路径
output_dir = Path('THP-1_analysis_plots')
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
    if x.size < 3:  # 至少需要3个点才能计算协方差
        print("警告: 组数据点少于3个，无法绘制置信椭圆。")
        return
        
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
    # 假设CSV文件名是 THP-1_PCA.csv
    df = pd.read_csv('THP-1_PCA.csv', encoding='gbk') 
except FileNotFoundError:
    print("错误: 'THP-1_PCA.csv' 未找到。请确保该文件与脚本在同一目录下。")
    exit()

# 根据您提供的文件，精确修正列名映射
column_mapping = {
    'Isolate_Name': 'Isolate_Name',
    '2H内化率(%)': 'Internalization_Rate', # 修正自THP-1_PCA.csv，名称可能不同
    'Intracellular bacteria_fold_change_12h': 'Intracellular_bacteria_fold_change_12h',
    'Intracellular bacteria_fold_change_24h': 'Intracellular_bacteria_fold_change_24h',
    'Intracellular bacteria_fold_change_48h': 'Intracellular_bacteria_fold_change_48h',
    'Live%_6h': 'Live_percent_6h',
    'Live%_12h': 'Live_percent_12h',
    'Live%_24h': 'Live_percent_24h',
    'Live%_48h': 'Live_percent_48h',
    'IL1a_6h': 'IL1a_6h', 'IL1b_6h': 'IL1b_6h', 'IL6_6h': 'IL6_6h',
    'IL8_6h': 'IL8_6h', 'IL10_6h': 'IL10_6h', 'TNFa_6h': 'TNFa_6h',
    'IL1a_12h': 'IL1a_12h', 'IL1b_12h': 'IL1b_12h', 'IL6_12h': 'IL6_12h',
    'IL8_12h': 'IL8_12h', 'IL10_12h': 'IL10_12h', 'TNFa_12h': 'TNFa_12h',
    'IL1a_24h': 'IL1a_24h', 'IL1b_24h': 'IL1b_24h', 'IL6_24h': 'IL6_24h',
    'IL8_24h': 'IL8_24h', 'IL10_24h': 'IL10_24h', 'TNFa_24h': 'TNFa_24h',
    'IL1a_48h': 'IL1a_48h', 'IL1b_48h': 'IL1b_48h', 'IL6_48h': 'IL6_48h',
    'IL8_48h': 'IL8_48h', 'IL10_48h': 'IL10_48h', 'TNFa_48h': 'TNFa_48h'
}
df = df.rename(columns=column_mapping)

# 修正分组信息提取逻辑，使用正则表达式来分割组名和编号
def parse_isolate_name(name):
    # 根据THP-1_PCA.csv文件的Isolate_Name格式，例如 'Low_probability01'
    match = re.match(r'([a-zA-Z_]+)(\d+)', name)
    if match:
        return match.groups()
    return name, None

df[['Group', 'SubjectID']] = df['Isolate_Name'].apply(lambda x: pd.Series(parse_isolate_name(x)))


# 准备数值型数据和处理缺失值
features = df.columns.drop(['Isolate_Name', 'Group', 'SubjectID'])
# 创建一个副本，用于PCA和后续分析，并处理其中的NaN值
df_no_nan = df.copy()
if df_no_nan[features].isnull().values.any():
    print("警告: 数据中检测到缺失值(NaN)。将使用每列的平均值进行填充。")
    df_no_nan[features] = df_no_nan[features].fillna(df_no_nan[features].mean())

# X 用于 PCA 分析
X = df_no_nan[features]


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
pca_df['Group'] = df_no_nan['Group']

explained_variance = pca.explained_variance_ratio_
print(f"PC1 解释的方差: {explained_variance[0]:.2%}")
print(f"PC2 解释的方差: {explained_variance[1]:.2%}")

# 2.4 PERMANOVA 统计分析
numpy_distance_matrix = pairwise_distances(X_scaled, metric='euclidean')
numpy_distance_matrix = (numpy_distance_matrix + numpy_distance_matrix.T) / 2.0
skbio_distance_matrix = DistanceMatrix(numpy_distance_matrix, ids=df_no_nan.index)

# 2.4.1 总体 PERMANOVA
print("\n--- 总体 PERMANOVA 统计结果 ---")
permanova_results_overall = permanova(skbio_distance_matrix, df_no_nan['Group'])
permanova_p_value_overall = permanova_results_overall['p-value']
print(permanova_results_overall)

# 2.4.2 两两比较 PERMANOVA (此数据集只有两组，所以总检验即为两两比较)
print("\n--- 两组 PERMANOVA 统计结果 ---")
groups = df_no_nan['Group'].unique()
if len(groups) == 2:
    group1, group2 = groups[0], groups[1]
    pairwise_p_value = permanova(skbio_distance_matrix, df_no_nan['Group'])['p-value']
    print(f"PERMANOVA for {group1} vs {group2}: p-value = {pairwise_p_value:.4f}")


# --- 3. PCA 可视化 (带置信椭圆和统计标注) ---

print("\n--- 正在生成并保存PCA图表 ---")
# 增大图表尺寸
fig, ax = plt.subplots(figsize=(15, 13))

# 绘制散点图
sns.scatterplot(
    x='PC1', y='PC2', hue='Group', style='Group',
    data=pca_df, s=150, alpha=0.8, palette='viridis', ax=ax
)

# 为每个组添加95%置信椭圆
colors = sns.color_palette('viridis', n_colors=len(groups))
color_map = dict(zip(groups, colors))

for group in groups:
    group_data = pca_df[pca_df['Group'] == group]
    confidence_ellipse(group_data['PC1'], group_data['PC2'], ax,
                       edgecolor=color_map[group], linewidth=2)

ax.set_title('不同感染概率组的PCA分析', fontsize=18)
ax.set_xlabel(f'主成分 1 ({explained_variance[0]:.2%})', fontsize=14)
ax.set_ylabel(f'主成分 2 ({explained_variance[1]:.2%})', fontsize=14)
ax.legend(title='分组', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)

# 在图上标注统计结果
stats_text_overall = f"总体 PERMANOVA: p = {permanova_p_value_overall:.4f}"
plt.text(0.02, 0.02, stats_text_overall, transform=ax.transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment='left',
         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

# 自动调整坐标轴范围，确保所有点和椭圆都可见
plt.autoscale(enable=True, axis='both', tight=True)
ax.margins(0.15) # 增加15%的边距

# 保存PCA图
pca_plot_filename = output_dir / 'THP-1_PCA_with_Ellipse_and_Stats.pdf'
plt.savefig(pca_plot_filename, format='pdf', bbox_inches='tight')
plt.close()
print(f"PCA图表已保存至: {pca_plot_filename}")


# --- 4. 各变量的统计分析与可视化 ---

print("\n--- 正在对非时间序列指标进行分析和绘图 ---")
# 定义单点指标
single_features = ['Internalization_Rate']
for var in single_features:
    print(f"\n--- {var} 的分析结果 ---")
    
    group_data = df_no_nan[['Group', var]].dropna()
    low_prob = group_data[group_data['Group'] == 'Low_probability'][var]
    high_prob = group_data[group_data['Group'] == 'High_probability'][var]
    
    ttest_result = pg.ttest(x=low_prob, y=high_prob, paired=False)
    p_val = ttest_result['p-val'][0]
    
    print("独立样本 T-检验:")
    print(ttest_result)

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_no_nan, x='Group', y=var, order=groups, palette='pastel', showfliers=False)
    sns.stripplot(data=df_no_nan, x='Group', y=var, order=groups, hue='Group', palette='pastel', jitter=0.2, alpha=0.6, s=12, marker='o')
    plt.title(f'{var} 在不同组中的比较', fontsize=16)
    plt.ylabel(var, fontsize=12)
    plt.xlabel('分组', fontsize=12)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='分组')

    stats_text = "独立样本T检验 p值:\n"
    sig_symbol = 'ns'
    if p_val < 0.001: sig_symbol = '***'
    elif p_val < 0.01: sig_symbol = '**'
    elif p_val < 0.05: sig_symbol = '*'
    stats_text += f"Low vs High: p={p_val:.4f} ({sig_symbol})\n"
    
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.7))

    plt.tight_layout()
    plot_filename = output_dir / f'{var}_boxplot.pdf'
    plt.savefig(plot_filename, format='pdf')
    plt.close()

# 定义所有时间点
time_points_all = ['6h', '12h', '24h', '48h']

# 修正：定义时间序列指标，包括细胞因子，并指定它们的时间点
# 确保每个单独的细胞因子也能生成时间序列图
time_series_vars = {
    'Intracellular_bacteria_fold_change': ['12h', '24h', '48h'], # 根据您的文件，这个只有12h,24h,48h
    'Live_percent': time_points_all,
    'IL1a': time_points_all,
    'IL1b': time_points_all,
    'IL6': time_points_all,
    'IL8': time_points_all,
    'IL10': time_points_all,
    'TNFa': time_points_all
}

print("\n--- 正在对时间序列指标进行分析和绘图 ---")
for var, tps in time_series_vars.items():
    plt.figure(figsize=(10, 7))
    
    # 筛选并重塑数据，以便在同一张图上绘制所有时间点
    cols_to_melt = [f'{var}_{tp}' for tp in tps if f'{var}_{tp}' in df_no_nan.columns]
    
    if not cols_to_melt:
        print(f"警告: 找不到 '{var}' 的时间序列数据 (期望列: {', '.join([f'{var}_{tp}' for tp in tps])})，跳过此变量的绘图。")
        continue
    
    melted_df = df_no_nan.melt(id_vars=['Group'], value_vars=cols_to_melt, var_name='TimePoint', value_name='Value')
    melted_df['TimePoint'] = melted_df['TimePoint'].apply(lambda x: x.split('_')[-1])

    print(f"\n--- {var} 的分析结果 ---")
    stats_text = "独立样本T检验 p值:\n"
    
    for tp in melted_df['TimePoint'].unique():
        low_prob = melted_df[(melted_df['TimePoint'] == tp) & (melted_df['Group'] == 'Low_probability')]['Value'].dropna()
        high_prob = melted_df[(melted_df['TimePoint'] == tp) & (melted_df['Group'] == 'High_probability')]['Value'].dropna()
        
        if len(low_prob) > 1 and len(high_prob) > 1:
            ttest_result = pg.ttest(x=low_prob, y=high_prob, paired=False)
            p_val = ttest_result['p-val'][0]
            sig_symbol = 'ns'
            if p_val < 0.001: sig_symbol = '***'
            elif p_val < 0.01: sig_symbol = '**'
            elif p_val < 0.05: sig_symbol = '*'
            stats_text += f"  {tp}: p={p_val:.4f} ({sig_symbol})\n"
            print(f"独立样本 T-检验 at {tp}:")
            print(ttest_result)
    
    sns.boxplot(data=melted_df, x='TimePoint', y='Value', hue='Group', palette='pastel', showfliers=False)
    sns.stripplot(data=melted_df, x='TimePoint', y='Value', hue='Group', palette='pastel', jitter=0.2, alpha=0.6, s=12, marker='o', dodge=True)

    plt.title(f'{var} 在不同时间点和组中的比较', fontsize=16)
    plt.ylabel(f'{var} 值', fontsize=12)
    plt.xlabel('时间点', fontsize=12)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='分组')
    
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.7))

    plt.tight_layout()
    plot_filename = output_dir / f'{var}_time_series_boxplot.pdf'
    plt.savefig(plot_filename, format='pdf')
    plt.close()


# --- 5. 组合细胞因子按时间点进行分析与绘图 ---
# 这部分生成的是一个时间点上所有细胞因子的箱线图 (例如 cytokines_at_6h_boxplot.pdf)
print("\n--- 正在按时间点组合细胞因子进行分析和绘图 ---")

# 定义要组合的细胞因子列表 (前缀)
cytokines_to_group = ['IL1a', 'IL1b', 'IL6', 'IL8', 'IL10', 'TNFa']

for tp in time_points_all: # 遍历所有时间点
    print(f"\n--- 正在分析时间点 {tp} 的细胞因子 ---")
    
    # 筛选当前时间点的细胞因子列 (完整列名，如 IL1a_6h)
    cols_to_melt = [f'{cytokine}_{tp}' for cytokine in cytokines_to_group]
    
    # 检查所有列是否存在
    if not all(col in df_no_nan.columns for col in cols_to_melt):
        print(f"警告: 时间点 {tp} 的部分细胞因子列缺失 (期望列: {', '.join(cols_to_melt)})，跳过此时间点的绘图。")
        continue

    # 准备用于绘图的melted DataFrame
    melted_df_cytokines = df_no_nan.melt(id_vars=['Group'], value_vars=cols_to_melt,
                                          var_name='Cytokine', value_name='Value')
    
    # 移除列名中的时间点后缀，只保留细胞因子名称，用于X轴标签
    melted_df_cytokines['Cytokine'] = melted_df_cytokines['Cytokine'].apply(lambda x: x.split('_')[0])
    
    plt.figure(figsize=(12, 8))
    
    # 绘制箱线图
    ax = sns.boxplot(data=melted_df_cytokines, x='Cytokine', y='Value', hue='Group', 
                     palette='pastel', showfliers=False)
    # 绘制散点图
    sns.stripplot(data=melted_df_cytokines, x='Cytokine', y='Value', hue='Group', 
                  palette='pastel', jitter=0.2, alpha=0.6, s=12, marker='o', dodge=True)

    # 对每个细胞因子进行独立 t 检验并标注结果
    stats_text = "独立样本T检验 p值:\n"
    # 这里遍历的是细胞因子名称，不是完整的列名
    for cytokine in cytokines_to_group: 
        # 再次从原始 df_no_nan 中筛选特定时间点+细胞因子的数据进行t检验
        col_name_full = f'{cytokine}_{tp}'
        if col_name_full in df_no_nan.columns: # 确保列存在
            low_prob = df_no_nan[df_no_nan['Group'] == 'Low_probability'][col_name_full].dropna()
            high_prob = df_no_nan[df_no_nan['Group'] == 'High_probability'][col_name_full].dropna()
            
            if len(low_prob) > 1 and len(high_prob) > 1:
                ttest_result = pg.ttest(x=low_prob, y=high_prob, paired=False)
                p_val = ttest_result['p-val'][0]
                sig_symbol = 'ns'
                if p_val < 0.001: sig_symbol = '***'
                elif p_val < 0.01: sig_symbol = '**'
                elif p_val < 0.05: sig_symbol = '*'
                stats_text += f"  {cytokine}: p={p_val:.4f} ({sig_symbol})\n"
                print(f"独立样本 T-检验 for {cytokine} at {tp}:")
                print(ttest_result)
            else:
                stats_text += f"  {cytokine}: 数据不足，无法进行T检验\n"
                print(f"警告: 时间点 {tp} 的 {cytokine} 数据不足，无法进行T检验。")
        else:
            stats_text += f"  {cytokine}: 对应列 '{col_name_full}' 不存在\n"
            print(f"警告: 列 '{col_name_full}' 不存在于数据中。")


    plt.title(f'{tp} 时间点细胞因子在不同组中的比较', fontsize=16)
    plt.ylabel('细胞因子值', fontsize=12)
    plt.xlabel('细胞因子', fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='分组')

    plt.text(0.98, 0.98, stats_text, transform=ax.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.7))

    plt.tight_layout()
    plot_filename = output_dir / f'cytokines_at_{tp}_boxplot.pdf'
    plt.savefig(plot_filename, format='pdf')
    plt.close()

print(f"\n分析完成。所有图表已保存至 '{output_dir}' 文件夹。")