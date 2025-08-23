# -*- coding: utf-8 -*-
"""
菌株COG功能分群分析完整版
功能：GMM分群 + 分群菌株列表输出 + Top差异COG热图
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ====================
# 1. 数据预处理
# ====================
# 读取数据并转置（行=菌株，列=COG）
df = pd.read_csv("4147LP_merged_COG2024_for_CNS_known_all.csv", index_col=0).T
print(f"数据维度：{df.shape} (菌株数={df.shape[0]}, COG功能数={df.shape[1]})")

# 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ====================
# 2. GMM分群与结果保存
# ====================
# PCA降维
pca = PCA(n_components=2)
pca_scores = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_

# GMM分群
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
clusters = gmm.fit_predict(pca_scores)
silhouette = silhouette_score(pca_scores, clusters)

# 保存分群结果（菌株ID + 分群标签）
cluster_df = pd.DataFrame({
    'Strain': df.index,
    'Cluster': clusters
})
os.makedirs("results", exist_ok=True)
cluster_df.to_csv("results/cluster_strains.csv", index=False)

# ====================
# 3. 分群可视化
# ====================
plt.figure(figsize=(10, 8))
sns.scatterplot(x=pca_scores[:,0], y=pca_scores[:,1], hue=clusters,
                palette={0: '#1f77b4', 1: '#ff7f0e'},  # 指定分群颜色
                s=60, edgecolor='w', alpha=0.8)
plt.title(f"GMM Clustering on PCA\n(Silhouette = {silhouette:.2f})")
plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
plt.savefig("results/pca_gmm_clusters.pdf", bbox_inches='tight', format='pdf')
plt.close()

# ====================
# 4. 差异COG功能分析
# ====================
p_values = []
effect_sizes = []

for cog in df.columns:
    group0 = df.loc[clusters == 0, cog]
    group1 = df.loc[clusters == 1, cog]
    
    # 跳过无效分组
    if len(group0) == 0 or len(group1) == 0:
        p_values.append(np.nan)
        effect_sizes.append(np.nan)
        continue
    
    # Mann-Whitney U检验
    try:
        stat, p = mannwhitneyu(group0, group1, alternative='two-sided')
    except ValueError:  # 数据完全相同的情况
        p = 1.0
    
    # 计算Cohen's d效应量
    n0, n1 = len(group0), len(group1)
    pooled_std = np.sqrt(((n0-1)*group0.var() + (n1-1)*group1.var()) / (n0 + n1 - 2))
    d = (group1.mean() - group0.mean()) / pooled_std if pooled_std != 0 else 0.0
    
    p_values.append(p)
    effect_sizes.append(d)

# 多重检验校正
_, pvals_adj, _, _ = multipletests(p_values, method='fdr_bh')

# 构建差异结果表
diff_results = pd.DataFrame({
    'COG': df.columns,
    'p_value': p_values,
    'p_adj': pvals_adj,
    'effect_size': effect_sizes,
    'cluster0_mean': df.loc[clusters == 0].mean().values,
    'cluster1_mean': df.loc[clusters == 1].mean().values
}).sort_values('effect_size', key=abs, ascending=False)

# 筛选Top10差异COG
sig_cogs = diff_results[
    (diff_results['p_adj'] < 0.05) &
    (abs(diff_results['effect_size']) > 0.5)
].head(10)

# ====================
# 5. 差异COG热图绘制
# ====================
if not sig_cogs.empty:
    # 按分群排序菌株（左侧Cluster0，右侧Cluster1）
    sorted_indices = np.concatenate([
        np.where(clusters == 0)[0],  # Cluster0的索引
        np.where(clusters == 1)[0]   # Cluster1的索引
    ])
    sorted_data = df.iloc[sorted_indices]
    
    # 提取Top10 COG数据并转置（行=COG，列=菌株）
    plot_data = sorted_data[sig_cogs['COG']].T
    
    # 绘制热图
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        plot_data,
        cmap="viridis",
        yticklabels=True,
        xticklabels=False,
        robust=True,
        cbar_kws={'label': 'Normalized Counts'}
    )
    
    # 添加分群分隔线
    cluster0_count = np.sum(clusters == 0)
    plt.axvline(x=cluster0_count, color='red', linestyle='--', linewidth=2)
    
    plt.title("Top 10 Differential COGs (Cluster0 vs Cluster1)")
    plt.savefig("results/diff_heatmap.pdf", bbox_inches='tight', format='pdf')
    plt.close()
else:
    print("警告：未找到显著差异的COG功能！")

# ====================
# 6. 结果汇总输出
# ====================
# 保存统计结果
sig_cogs.to_csv("results/significant_cogs.csv", index=False)

# 生成分析报告
with open("results/analysis_summary.txt", "w") as f:
    f.write(f"""=== 菌株COG功能分群分析报告 ===
* 分群结果：
  - Cluster0菌株数: {np.sum(clusters == 0)}
  - Cluster1菌株数: {np.sum(clusters == 1)}
  - 分群菌株列表见: cluster_strains.csv
  
* 主成分分析：
  - PC1解释方差: {explained_var[0]*100:.1f}%
  - PC2解释方差: {explained_var[1]*100:.1f}%
  - 轮廓系数: {silhouette:.2f}
  
* 差异COG功能：
  - 显著差异COG数量: {len(sig_cogs)}个
  - Top10差异COG列表: {', '.join(sig_cogs['COG'].tolist()) if not sig_cogs.empty else '无'}
  - 详细信息见: significant_cogs.csv
""")

print("分析完成！结果已保存至 results/ 目录")