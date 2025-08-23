# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve, average_precision_score,
    brier_score_loss
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# PDF 字体设置，确保 Illustrator 可编辑
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
import shap
import joblib
import logging
from collections import defaultdict
import gc
import traceback
import uuid
import warnings
from sklearn.exceptions import ConvergenceWarning

# Filter warnings for cleaner logs
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ================= 配置参数 =================
CONFIG = {
    "output_dir_base": "all_4147LP_kfold_results_final", # 更新输出目录名
    "n_splits_cv": 5,
    "random_state": 42,
    "shap": {
        "background_samples": 100,
        "top_n_interaction_features": 10,
        "decision_plot_n_samples": 20,
        "shap_analysis_models": ['XGBoost', 'Random Forest', 'Logistic Regression', 'Gradient Boosting', 'Decision Tree', 'LightGBM', 'ElasticNet Logistic Regression', 'Linear Discriminant Analysis'], # 扩展SHAP分析模型列表
        "max_shap_samples_agg": 9999,
        "heatmap_sample_size": 9999
    },
    "filter": {
        "min_freq": 0.05,
        "max_freq": 0.95
    },
    "top_n_features": [20, 50],
    "bootstrap_ci": {
        "enabled": True,
        "n_iterations": 1000,
        "confidence_level": 0.95
    }
}
# ================= 初始化日志 =================
def setup_logging(output_dir, log_file_name='analysis.log'):
    """设置日志记录器，同时输出到文件和终端"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, log_file_name)
    # 创建更独特的 logger 名称避免冲突
    logger_name = f"log_{os.path.basename(output_dir)}_{log_file_name.replace('.', '_')}_{uuid.uuid4().hex[:6]}"
    logger = logging.getLogger(logger_name)

    # 如果 logger 已存在且有处理器，则不再添加，避免重复日志
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 控制台处理器
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # 防止日志向上传播到根 logger
        logger.propagate = False

    return logger

# ================= 初始化设置 =================
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# ================= 加载数据 =================
def load_data(logger):
    """加载泛基因组和表型数据，合并并清理"""
    logger.info("开始加载数据...")
    try:
        pan_genome_df = pd.read_excel('all_4147LP.xlsx', index_col=0).T
        logger.info(f"泛基因组数据加载成功，形状: {pan_genome_df.shape}")
    except FileNotFoundError:
        logger.error("错误：找不到泛基因组文件 'all_4147LP.xlsx'")
        raise
    except Exception as e:
        logger.error(f"加载泛基因组数据时出错: {e}")
        raise

    try:
        metadata_df = pd.read_excel('metadata_all_4147LP_min.xlsx')
        metadata_df['菌株名'] = metadata_df['菌株名'].astype(str)
        metadata_df = metadata_df[['菌株名', '菌株表型']].set_index('菌株名')
        logger.info(f"表型数据加载成功，形状: {metadata_df.shape}")
    except FileNotFoundError:
        logger.error("错误：找不到表型文件 'metadata_all_4147LP_min.xlsx'")
        raise
    except Exception as e:
        logger.error(f"加载表型数据时出错: {e}")
        raise

    # 确保索引是字符串类型以便合并
    metadata_df.index = metadata_df.index.astype(str)
    pan_genome_df.index = pan_genome_df.index.astype(str)

    merged_df = pd.merge(metadata_df, pan_genome_df, left_index=True, right_index=True, how='inner')
    logger.info(f"数据合并后形状: {merged_df.shape}")

    if merged_df.empty:
        logger.error("错误：元数据和泛基因组数据合并后为空，请检查索引是否匹配。")
        raise ValueError("数据合并后为空")


    merged_df['菌株表型'] = merged_df['菌株表型'].astype(str).str.strip()
    valid_phenotypes = ['Clinical', 'Environmental']
    initial_rows = merged_df.shape[0]
    merged_df = merged_df[merged_df['菌株表型'].isin(valid_phenotypes)].copy()
    logger.info(f"过滤非目标表型后，保留 {merged_df.shape[0]} 行 (移除 {initial_rows - merged_df.shape[0]} 行)")

    merged_df['Label'] = merged_df['菌株表型'].map({'Clinical': 1, 'Environmental': 0})
    if merged_df['Label'].isnull().any():
        nan_count = merged_df['Label'].isnull().sum()
        logger.warning(f"发现 {nan_count} 个 NaN 标签，移除这些行")
        merged_df = merged_df.dropna(subset=['Label']).copy()

    if merged_df.empty:
        logger.error("错误：过滤和移除NaN标签后数据为空。")
        raise ValueError("数据为空")

    label_counts = merged_df['Label'].value_counts()
    if len(label_counts) < 2:
        logger.error(f"错误：数据仅包含单一标签 {label_counts.index.tolist()}，无法进行分类")
        raise ValueError("数据必须包含两种标签")
    if label_counts.min() < CONFIG["n_splits_cv"]:
        logger.error(f"错误：最少类别的样本数 ({label_counts.min()}) 少于 K-Fold 折数 ({CONFIG['n_splits_cv']})")
        raise ValueError("样本数不足以进行分层 K 折")

    logger.info(f"数据加载和清理完成，保留 {merged_df.shape[0]} 行。标签分布:\n{label_counts}")
    return merged_df

# ================= 数据预处理 =================
def preprocess_data_full(merged_df, logger):
    """对完整数据集进行特征过滤"""
    logger.info("开始对完整数据集进行数据预处理...")
    X = merged_df.drop(['菌株表型', 'Label'], axis=1)
    y = merged_df['Label'].astype(int)

    if X.isnull().any().any():
        logger.warning("特征矩阵包含 NaN 值，用 0 填充以计算频率")
        X = X.fillna(0) # 直接修改X

    # 确保所有特征列都是数值类型
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_cols.empty:
        logger.warning(f"发现非数值特征列: {non_numeric_cols.tolist()}. 尝试转换为数值类型，无法转换的将填充为0。")
        for col in non_numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0) # 填充转换失败产生的NaN

    gene_freq = X.mean(axis=0)
    # 假设 CONFIG 已经定义并且包含 filter 键和 min_freq, max_freq
    valid_genes = gene_freq.between(CONFIG["filter"]["min_freq"], CONFIG["filter"]["max_freq"])

    if not valid_genes.any():
        logger.error("错误：根据频率过滤后没有剩余特征")
        raise ValueError("过滤后无特征")

    X_filtered = X.loc[:, valid_genes]
    feature_names = X_filtered.columns.tolist()

    # 检查特征名中是否有特殊字符，可能影响后续处理（如绘图）
    invalid_chars = ['<', '>', '[', ']', '{', '}', ':', ',']
    cleaned_feature_names = []
    renamed_count = 0
    for name in feature_names:
        cleaned_name = str(name)
        for char in invalid_chars:
            cleaned_name = cleaned_name.replace(char, '_')
        if cleaned_name != str(name):
            renamed_count += 1
        cleaned_feature_names.append(cleaned_name)

    if renamed_count > 0:
        logger.warning(f"已清理 {renamed_count} 个特征名中的特殊字符，以避免后续错误。")
        X_filtered.columns = cleaned_feature_names
        feature_names = cleaned_feature_names

    logger.info(f"完整数据集过滤后形状: {X_filtered.shape}, 特征数量: {len(feature_names)}")
    logger.info("完整数据集预处理完成")

    X_np = X_filtered.values
    y_np = y.values
    gc.collect()

    # 确保 CONFIG["output_dir_base"] 存在且是有效的路径
    output_path = os.path.join(CONFIG["output_dir_base"], 'training_feature_names.txt')
    os.makedirs(CONFIG["output_dir_base"], exist_ok=True) # 确保输出目录存在

    with open(output_path, 'w', encoding='utf-8') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    logger.info(f"完整的训练特征列表已保存至: {output_path}")

    return X_np, y_np, feature_names, X_filtered.index

# ================= 单折绘图函数 =================
def plot_roc_curve(y_true, y_proba, title, filename, n_bootstrap=1000):
    """绘制单折 ROC 曲线，带 95% 置信区间"""
    current_logger = logging.getLogger(logging.getLogger().name)
    plt.figure(figsize=(8, 6))
    try:
        # 检查y_proba中是否有无效值
        valid_indices = ~np.isnan(y_proba)
        y_true_valid = y_true[valid_indices]
        y_proba_valid = y_proba[valid_indices]

        if len(y_true_valid) < 2 or len(np.unique(y_true_valid)) < 2:
             current_logger.warning(f"无法为 {title} 生成 ROC 曲线，有效数据不足或标签单一。")
             plt.plot([0, 1], [0, 1], 'k--', label='Chance')
             plt.title(f"{title}\n(Insufficient Data)")
             plt.xlabel('False Positive Rate')
             plt.ylabel('True Positive Rate')
             plt.legend()
             plt.savefig(filename)
             return

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for i in range(n_bootstrap):
            indices = np.random.choice(len(y_true_valid), len(y_true_valid), replace=True)
            y_true_sample = y_true_valid[indices]
            y_proba_sample = y_proba_valid[indices]

            if len(np.unique(y_true_sample)) < 2:
                continue # 跳过只有单一类别的 bootstrap 样本

            try:
                fpr, tpr, _ = roc_curve(y_true_sample, y_proba_sample)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(roc_auc_score(y_true_sample, y_proba_sample))
            except ValueError as roc_e:
                 current_logger.debug(f"Bootstrapped ROC iteration failed: {roc_e}")
                 continue


        if not tprs:
            current_logger.warning(f"无法为 {title} 生成有效的 bootstrapped ROC 数据")
            # 绘制基础ROC曲线 (无CI)
            fpr_base, tpr_base, _ = roc_curve(y_true_valid, y_proba_valid)
            auc_base = roc_auc_score(y_true_valid, y_proba_valid)
            plt.plot(fpr_base, tpr_base, color='darkorange', label=f'ROC (AUC = {auc_base:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title(f"{title}\n(Base Curve - No CI)")
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.savefig(filename)
            return

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        lower_auc = np.percentile(aucs, 2.5)
        upper_auc = np.percentile(aucs, 97.5)

        plt.plot(mean_fpr, mean_tpr, color='darkorange', label=f'Mean ROC (AUC = {mean_auc:.2f}, 95% CI [{lower_auc:.2f}-{upper_auc:.2f}])')
        tprs_upper = np.percentile(tprs, 97.5, axis=0)
        tprs_lower = np.percentile(tprs, 2.5, axis=0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='95% CI')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        current_logger.info(f"ROC 曲线保存至: {filename}")
    except Exception as e:
        current_logger.error(f"绘制 ROC 曲线 ({title}) 出错: {e}", exc_info=True)
    finally:
        plt.close()

def plot_pr_curve(y_true, y_proba, title, filename, n_bootstrap=1000):
    """绘制单折 PR 曲线，带 95% 置信区间"""
    current_logger = logging.getLogger(logging.getLogger().name)
    plt.figure(figsize=(8, 6))
    try:
        # 检查y_proba中是否有无效值
        valid_indices = ~np.isnan(y_proba)
        y_true_valid = y_true[valid_indices]
        y_proba_valid = y_proba[valid_indices]

        if len(y_true_valid) < 2 or len(np.unique(y_true_valid)) < 2:
             current_logger.warning(f"无法为 {title} 生成 PR 曲线，有效数据不足或标签单一。")
             baseline = np.mean(y_true_valid) if len(y_true_valid) > 0 else 0.5
             plt.plot([0, 1], [baseline, baseline], 'k--', label=f'Baseline ({baseline:.2f})')
             plt.title(f"{title}\n(Insufficient Data)")
             plt.xlabel('Recall')
             plt.ylabel('Precision')
             plt.legend()
             plt.savefig(filename)
             return

        precisions_interp = []
        aps = [] # Average Precision scores
        mean_recall = np.linspace(0, 1, 100)

        for i in range(n_bootstrap):
            indices = np.random.choice(len(y_true_valid), len(y_true_valid), replace=True)
            y_true_sample = y_true_valid[indices]
            y_proba_sample = y_proba_valid[indices]

            if len(np.unique(y_true_sample)) < 2:
                continue

            try:
                precision, recall, _ = precision_recall_curve(y_true_sample, y_proba_sample)
                ap_score = average_precision_score(y_true_sample, y_proba_sample)

                # 对 recall 进行排序，确保插值正确
                sorted_idx = np.argsort(recall)
                recall_sorted = recall[sorted_idx]
                precision_sorted = precision[sorted_idx]

                # 确保 recall 从 0 开始，precision 从 1 (或对应值) 开始
                if recall_sorted[0] > 0:
                    recall_sorted = np.concatenate([[0], recall_sorted])
                    precision_sorted = np.concatenate([[precision_sorted[0]], precision_sorted]) # 通常在recall=0时，precision是未定义的，但插值需要点，这里用第一个实际值
                if recall_sorted[-1] < 1:
                     recall_sorted = np.concatenate([recall_sorted, [1]])
                     precision_sorted = np.concatenate([precision_sorted, [np.mean(y_true_sample)]]) # 在recall=1时，precision是该类的基线概率

                interp_precision = np.interp(mean_recall, recall_sorted, precision_sorted)
                precisions_interp.append(interp_precision)
                aps.append(ap_score)
            except ValueError as pr_e:
                 current_logger.debug(f"Bootstrapped PR iteration failed: {pr_e}")
                 continue


        if not precisions_interp:
            current_logger.warning(f"无法为 {title} 生成有效的 bootstrapped PR 数据")
            # 绘制基础PR曲线 (无CI)
            precision_base, recall_base, _ = precision_recall_curve(y_true_valid, y_proba_valid)
            ap_base = average_precision_score(y_true_valid, y_proba_valid)
            plt.plot(recall_base, precision_base, color='blue', label=f'PR (AP = {ap_base:.2f})')
            baseline = np.mean(y_true_valid)
            plt.plot([0, 1], [baseline, baseline], 'k--', label=f'Baseline ({baseline:.2f})')
            plt.title(f"{title}\n(Base Curve - No CI)")
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            plt.savefig(filename)
            return

        mean_precision = np.mean(precisions_interp, axis=0)
        mean_ap = np.mean(aps)
        std_ap = np.std(aps)
        lower_ap = np.percentile(aps, 2.5)
        upper_ap = np.percentile(aps, 97.5)

        plt.plot(mean_recall, mean_precision, color='blue', label=f'Mean PR (AP = {mean_ap:.2f}, 95% CI [{lower_ap:.2f}-{upper_ap:.2f}])')
        precisions_lower = np.percentile(precisions_interp, 2.5, axis=0)
        precisions_upper = np.percentile(precisions_interp, 97.5, axis=0)
        plt.fill_between(mean_recall, precisions_lower, precisions_upper, color='grey', alpha=0.2, label='95% CI')

        baseline = np.mean(y_true_valid)
        plt.plot([0, 1], [baseline, baseline], 'k--', label=f'Baseline ({baseline:.2f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        current_logger.info(f"PR 曲线保存至: {filename}")
    except Exception as e:
        current_logger.error(f"绘制 PR 曲线 ({title}) 出错: {e}", exc_info=True)
    finally:
        plt.close()

def plot_confusion_matrix(cm, classes, title, filename):
    """绘制混淆矩阵"""
    current_logger = logging.getLogger(logging.getLogger().name)
    plt.figure(figsize=(8, 6))
    try:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 14})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(filename)
        current_logger.info(f"混淆矩阵保存至: {filename}")
    except Exception as e:
        current_logger.error(f"绘制混淆矩阵 ({title}) 出错: {e}", exc_info=True)
    finally:
        plt.close()

def plot_feature_importance(importance_series, title, filename, top_n=20):
    """绘制特征重要性条形图"""
    current_logger = logging.getLogger(logging.getLogger().name)
    plt.figure(figsize=(10, 6))
    try:
        n_plot = min(top_n, len(importance_series))
        if n_plot > 0:
            plot_data = importance_series.head(n_plot).sort_values(ascending=True) # 改为升序，方便横向条形图从上到下是重要性递减
            plot_data.index = plot_data.index.astype(str)
            plot_data.plot(kind='barh')
            # plt.gca().invert_yaxis() # 因为已经排序，不需要反转Y轴
            plt.title(f'{title} (Top {n_plot})')
            plt.xlabel('Importance Score')
            plt.ylabel('Gene')
            plt.tight_layout() # 调整布局防止标签重叠
            plt.savefig(filename)
            current_logger.info(f"特征重要性图保存至: {filename}")
        else:
             current_logger.warning(f"没有特征重要性数据可绘制 ({title})")

    except Exception as e:
        current_logger.error(f"绘制特征重要性图 ({title}) 出错: {e}", exc_info=True)
    finally:
        plt.close()
# ================= SHAP 绘图函数 =================
def plot_shap_summary(shap_values, features, feature_names, title, filename, max_display=20, plot_type="dot"):
    """绘制 SHAP Summary Plot (点图或条形图)"""
    current_logger = logging.getLogger(logging.getLogger().name)
    try:
        if not isinstance(shap_values, np.ndarray) or shap_values.ndim != 2:
            current_logger.warning(f"SHAP values not 2D for summary plot ({title}), skipping plot. Shape: {getattr(shap_values, 'shape', 'N/A')}")
            return
        # 对于summary plot，features可以是DataFrame或Numpy Array，但形状要匹配
        if features is None or features.shape[0] != shap_values.shape[0] or features.shape[1] != shap_values.shape[1]:
            current_logger.warning(f"Features shape mismatch with SHAP values for summary plot ({title}), skipping plot. SHAP shape: {shap_values.shape}, Features shape: {getattr(features, 'shape', 'N/A')}")
            return
        if not feature_names or len(feature_names) != shap_values.shape[1]:
            current_logger.warning(f"Feature names mismatch with SHAP values columns for summary plot ({title}), skipping plot. Feature name count: {len(feature_names) if feature_names else 0}, SHAP column count: {shap_values.shape[1]}")
            return

        # 显式创建 figure
        num_features_to_display = min(max_display, shap_values.shape[1])
        fig_height = max(6, num_features_to_display * 0.35)
        plt.figure(figsize=(10, fig_height))

        shap.summary_plot(shap_values, features, feature_names=feature_names,
                          show=False, max_display=num_features_to_display,
                          plot_type=plot_type)

        # 获取当前的 Figure 并调整
        fig = plt.gcf()
        fig.suptitle(title, y=1.02) # 使用 suptitle 并调整位置
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # 调整布局区域为标题留出空间

        plt.savefig(filename)
        current_logger.info(f"SHAP Summary Plot ({plot_type}) saved: {filename}")
    except Exception as e:
        current_logger.error(f"Error plotting SHAP summary plot ({title}): {e}", exc_info=True)
    finally:
        # 确保关闭的是当前创建的 figure
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        else:
            plt.close('all') # Fallback if figure retrieval failed

def plot_shap_heatmap_fold(explanation, title, filename, max_display=21):
    """绘制单折 SHAP Heatmap"""
    current_logger = logging.getLogger(logging.getLogger().name)
    plot_name = title
    current_logger.info(f"开始生成 {plot_name}")
    try:
        if not isinstance(explanation, shap.Explanation):
            current_logger.error(f"{plot_name} 的输入不是 SHAP Explanation 对象，跳过生成。")
            return
        shap_values = explanation.values
        feature_names = explanation.feature_names

        if not isinstance(shap_values, np.ndarray) or shap_values.ndim != 2:
            current_logger.error(f"{plot_name} 的 SHAP 值不是 2D 数组 (shape: {getattr(shap_values, 'shape', 'N/A')})，无法生成热图。")
            return
        if shap_values.shape[0] == 0 or shap_values.shape[1] == 0:
            current_logger.warning(f"{plot_name} 的 SHAP 值数组为空，跳过生成热图。")
            return
        if not feature_names or len(feature_names) != shap_values.shape[1]:
             # 如果 feature_names 是 None 或空列表，尝试从 explanation.data 获取 (如果是 DataFrame)
             if hasattr(explanation, 'data') and isinstance(explanation.data, pd.DataFrame):
                 feature_names = explanation.data.columns.tolist()
                 if len(feature_names) == shap_values.shape[1]:
                     current_logger.info(f"从 explanation.data 提取到特征名用于 {plot_name}")
                     explanation.feature_names = feature_names # 更新 explanation 对象
                 else:
                     current_logger.error(f"{plot_name} 的特征名称无效或数量不匹配 (预期: {shap_values.shape[1]}, 实际从data提取: {len(feature_names)})")
                     return
             else:
                 current_logger.error(f"{plot_name} 的特征名称无效或数量不匹配 (预期: {shap_values.shape[1]}, 实际: {len(feature_names)})")
                 return

        num_features_to_display = min(max_display, len(feature_names))
        num_samples = shap_values.shape[0]

        # 调整图形大小
        fig_height = max(10, num_samples * 0.1) # 根据样本数调整高度
        fig_width = max(12, num_features_to_display * 0.5) # 根据特征数调整宽度

        plt.figure(figsize=(fig_width, fig_height))

        # 尝试调用绘图函数
        shap.plots.heatmap(
            explanation,
            max_display=num_features_to_display,
            show=False
        )
        plt.title(f"{title}\n(Top {num_features_to_display} Features by mean |SHAP|)")
        plt.xlabel("Features (Ranked by mean |SHAP value|)")
        plt.ylabel(f"Samples (n={num_samples})")

        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # 为标题留空间
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        current_logger.info(f"{plot_name} 生成完成: {filename}")
    except Exception as e:
        current_logger.error(f"{plot_name} 生成失败: {str(e)}")
        current_logger.error(traceback.format_exc())
    finally:
        plt.close('all') # 确保关闭

def plot_shap_decision_fold(explanation, n_samples_to_plot, title, filename):
    """绘制单折 SHAP Decision Plot"""
    current_logger = logging.getLogger(logging.getLogger().name)
    try:
        if not isinstance(explanation, shap.Explanation):
            current_logger.warning(f"Input is not a SHAP Explanation object for decision plot ({title}), skipping plot.")
            return

        n_samples_available = explanation.shape[0]
        n_plot = min(n_samples_to_plot, n_samples_available)
        if n_plot == 0:
            current_logger.warning(f"No samples available for decision plot: {title}")
            return

        # 确保随机选择的索引不重复，即使 n_plot < n_samples_available
        sample_indices_idx = np.random.choice(np.arange(n_samples_available), n_plot, replace=False)
        explanation_sample = explanation[sample_indices_idx]

        base_value_input = explanation_sample.base_values
        # 处理 base_values 是数组的情况 (常见于 KernelExplainer 或多输出 TreeExplainer)
        if isinstance(base_value_input, np.ndarray):
            if base_value_input.ndim > 0 and base_value_input.size > 0:
                 base_value_to_use = base_value_input.mean() # 使用均值作为基准
                 current_logger.debug(f"Decision plot ({title}): Using mean base value {base_value_to_use:.4f} from array.")
            else:
                 current_logger.warning(f"Decision plot ({title}): Base value array is empty, using 0.")
                 base_value_to_use = 0
        elif isinstance(base_value_input, (float, int, np.number)):
            base_value_to_use = base_value_input
            current_logger.debug(f"Decision plot ({title}): Using scalar base value {base_value_to_use:.4f}.")
        else:
            current_logger.warning(f"Unexpected base value type for decision plot ({title}): {type(base_value_input)}, using 0 as fallback.")
            base_value_to_use = 0

        features_data = explanation_sample.data
        shap_values_data = explanation_sample.values
        feature_names_list = explanation_sample.feature_names

        if features_data is None or feature_names_list is None or shap_values_data is None:
            current_logger.warning(f"Explanation data, feature names, or SHAP values are missing for decision plot: {title}, skipping plot.")
            return
        if shap_values_data.shape[0] != n_plot or features_data.shape[0] != n_plot:
             current_logger.warning(f"Mismatch in sample dimensions for decision plot ({title}). SHAP: {shap_values_data.shape[0]}, Features: {features_data.shape[0]}, Expected: {n_plot}. Skipping plot.")
             return

        # 显式创建 figure
        plt.figure(figsize=(10, max(6, n_plot * 0.4)))

        shap.decision_plot(
            base_value=base_value_to_use,
            shap_values=shap_values_data,
            features=features_data,
            feature_names=feature_names_list,
            show=False
        )
        plt.title(f"{title}\n({n_plot} Random Validation Samples)")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        current_logger.info(f"SHAP Decision plot saved: {filename}")
    except Exception as e:
        current_logger.error(f"Error plotting SHAP decision plot ({title}): {e}", exc_info=True)
    finally:
        plt.close('all') # 确保关闭

def plot_shap_interactions_fold(interaction_values_3d, 
                                X_test_fold_sample_df, # 需要传入采样后的 X 数据
                                shap_values_fold_sample_2d, # 需要传入采样后的 SHAP 值
                                feature_names,
                                top_interactions,
                                title_prefix,
                                filename_prefix):
    """绘制单折 SHAP Interaction Plots（依赖图）"""
    current_logger = logging.getLogger(logging.getLogger().name)
    try:
        if not isinstance(interaction_values_3d, np.ndarray) or interaction_values_3d.ndim != 3:
            current_logger.warning(f"Cannot plot interactions for {title_prefix}, expected 3D interaction array, got {getattr(interaction_values_3d,'shape','N/A')}")
            return
        if not isinstance(shap_values_fold_sample_2d, np.ndarray) or shap_values_fold_sample_2d.ndim != 2:
            current_logger.warning(f"Cannot plot interactions for {title_prefix}, shap_values_fold_sample_2d is not 2D")
            return
        if not isinstance(X_test_fold_sample_df, pd.DataFrame):
             current_logger.warning(f"Cannot plot interactions for {title_prefix}, X_test_fold_sample_df is not a DataFrame")
             return
        if interaction_values_3d.shape[0] != shap_values_fold_sample_2d.shape[0] or interaction_values_3d.shape[0] != X_test_fold_sample_df.shape[0]:
             current_logger.warning(f"Sample size mismatch for interaction plot: Interactions({interaction_values_3d.shape[0]}), SHAP({shap_values_fold_sample_2d.shape[0]}), X({X_test_fold_sample_df.shape[0]})")
             return
        if interaction_values_3d.shape[1] != len(feature_names) or interaction_values_3d.shape[2] != len(feature_names):
             current_logger.warning(f"Feature dimension mismatch for interaction plot: Interactions({interaction_values_3d.shape[1:]}), Expected:({len(feature_names)}, {len(feature_names)})")
             return


        current_logger.info(f"Plotting {len(top_interactions)} interaction pairs for {title_prefix}...")
        plot_count = 0

        for feature1, feature2 in top_interactions:
            # 确保两个特征都在列表里
            if feature1 not in feature_names or feature2 not in feature_names:
                current_logger.warning(f"Skipping interaction plot: features {feature1}, {feature2} not found in feature list")
                continue

            # 绘制依赖图，传入特征名称
            try:
                plt.figure(figsize=(8, 6))
                shap.dependence_plot(
                    ind=feature1, # 主要特征
                    shap_values=shap_values_fold_sample_2d,
                    features=X_test_fold_sample_df,
                    feature_names=feature_names,
                    interaction_index=feature2, # 用于着色的交互特征
                    show=False
                )
                plt.title(f'{title_prefix}: SHAP Dependence\n{feature1} vs {feature2}')
                plt.tight_layout()

                # 安全文件名
                safe1 = str(feature1).replace('/', '_').replace('\\', '_').replace(':','_')
                safe2 = str(feature2).replace('/', '_').replace('\\', '_').replace(':','_')
                out_file = os.path.join(
                    os.path.dirname(filename_prefix), # 保存到 fold 目录
                    f"{os.path.basename(filename_prefix)}_{safe1}_vs_{safe2}_dependence.pdf" # 更新文件名
                )
                plt.savefig(out_file, dpi=300, bbox_inches='tight')
                plot_count += 1
            except Exception as dep_e:
                current_logger.error(f"Error plotting dependence for {feature1} vs {feature2}: {dep_e}", exc_info=True)
            finally:
                plt.close() # 关闭当前图形


        current_logger.info(f"Saved {plot_count} interaction (dependence) plots for {title_prefix}")

    except Exception as e:
        current_logger.error(f"Error during SHAP interaction plotting ({title_prefix}): {e}", exc_info=True)
    finally:
        plt.close('all') # 确保最终所有图形都关闭
# ================= Bootstrapped 曲线绘图 =================
def plot_bootstrapped_curve(y_true_agg, y_proba_agg, curve_type='ROC', n_bootstrap=1000, ax=None, title=None, color=None, label_prefix=None):
    """绘制带 bootstrapped 置信区间的曲线（ROC 或 PR）"""
    current_logger = logging.getLogger(logging.getLogger().name)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # 预处理：移除 NaN 概率值对应的样本
    valid_mask = ~np.isnan(y_proba_agg)
    y_true_valid = y_true_agg[valid_mask]
    y_proba_valid = y_proba_agg[valid_mask]
    n_samples_valid = len(y_true_valid)

    # 检查有效数据量和类别数
    if n_samples_valid < 2 or len(np.unique(y_true_valid)) < 2:
        current_logger.warning(f"数据不足或类别单一 ({n_samples_valid} valid samples), 无法绘制 {curve_type} 曲线 for {label_prefix}")
        if curve_type == 'ROC':
            ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Chance')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
        elif curve_type == 'PR':
            baseline = np.mean(y_true_valid) if n_samples_valid > 0 else 0.5
            ax.plot([0, 1], [baseline, baseline], linestyle='--', color='grey', label=f'Baseline ({baseline:.2f})')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
        ax.set_title(title or f'{curve_type} Curve (Insufficient Data)')
        ax.legend()
        ax.grid(True)
        # ax.set_aspect('equal', adjustable='box') # 可能导致 PR 曲线显示不佳
        plt.tight_layout()
        return ax # 返回轴对象

    # 初始化存储列表
    curve_values = []
    metric_values = []

    # 设置曲线参数
    if curve_type == 'ROC':
        base_x = np.linspace(0, 1, 101) # FPR
        plot_func = roc_curve
        metric_func = roc_auc_score
        xlabel = 'False Positive Rate'
        ylabel = 'True Positive Rate'
        metric_name = 'AUC'
        loc = 'lower right'
    elif curve_type == 'PR':
        base_x = np.linspace(0, 1, 101) # Recall
        metric_func = average_precision_score
        xlabel = 'Recall'
        ylabel = 'Precision'
        metric_name = 'Average Precision (AP)'
        loc = 'lower left'
    else:
        raise ValueError("curve_type must be 'ROC' or 'PR'")

    # Bootstrap 循环
    for i in range(n_bootstrap):
        try:
            indices = np.random.choice(n_samples_valid, n_samples_valid, replace=True)
            y_true_sample = y_true_valid[indices]
            y_proba_sample = y_proba_valid[indices]

            # 确保 bootstrap 样本有两类
            if len(np.unique(y_true_sample)) < 2:
                continue

            # 计算指标和曲线点
            metric = metric_func(y_true_sample, y_proba_sample)
            metric_values.append(metric)

            if curve_type == 'ROC':
                fpr, tpr, _ = plot_func(y_true_sample, y_proba_sample)
                interp_y = np.interp(base_x, fpr, tpr)
                interp_y[0] = 0.0 # 确保从原点开始
                curve_values.append(interp_y)
            elif curve_type == 'PR':
                precision, recall, _ = precision_recall_curve(y_true_sample, y_proba_sample)
                # 排序并确保完整范围以进行插值
                sorted_idx = np.argsort(recall)
                recall_sorted = recall[sorted_idx]
                precision_sorted = precision[sorted_idx]
                if recall_sorted[0] > 0:
                    recall_sorted = np.concatenate([[0], recall_sorted])
                    precision_sorted = np.concatenate([[precision_sorted[0]], precision_sorted])
                if recall_sorted[-1] < 1:
                    recall_sorted = np.concatenate([recall_sorted, [1]])
                    precision_sorted = np.concatenate([precision_sorted, [np.mean(y_true_sample)]]) # PR 在 R=1 时为基线
                interp_y = np.interp(base_x, recall_sorted, precision_sorted)
                curve_values.append(interp_y)

        except Exception as e:
            current_logger.debug(f"Bootstrap iteration {i} failed: {str(e)}")
            continue # 跳过失败的迭代

    # 检查是否有有效的 bootstrap 结果
    if not curve_values or not metric_values:
        current_logger.warning(f"无有效 bootstrap 样本，无法计算置信区间 for {label_prefix} {curve_type} curve.")
        # 尝试绘制基础曲线
        try:
            if curve_type == 'ROC':
                fpr, tpr, _ = plot_func(y_true_valid, y_proba_valid)
                metric_base = metric_func(y_true_valid, y_proba_valid)
                ax.plot(fpr, tpr, label=f'{label_prefix} ({metric_name} = {metric_base:.3f})', color=color)
                ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Chance')
            elif curve_type == 'PR':
                precision, recall, _ = precision_recall_curve(y_true_valid, y_proba_valid)
                metric_base = metric_func(y_true_valid, y_proba_valid)
                ax.plot(recall, precision, label=f'{label_prefix} ({metric_name} = {metric_base:.3f})', color=color)
                baseline = np.mean(y_true_valid)
                ax.plot([0, 1], [baseline, baseline], linestyle='--', color='grey', label=f'Baseline ({baseline:.2f})')
        except Exception as base_e:
             current_logger.error(f"无法绘制基础 {curve_type} 曲线: {base_e}")

        ax.set_title(title or f'{curve_type} Curve (No CI)')
        ax.legend(loc=loc)
        ax.grid(True)
        plt.tight_layout()
        return ax

    # 计算均值和置信区间
    mean_y = np.mean(curve_values, axis=0)
    lower_y = np.percentile(curve_values, 2.5, axis=0)
    upper_y = np.percentile(curve_values, 97.5, axis=0)
    mean_metric = np.mean(metric_values)
    lower_metric = np.percentile(metric_values, 2.5)
    upper_metric = np.percentile(metric_values, 97.5)

    # 准备标签
    label = f'{label_prefix} ({metric_name} = {mean_metric:.3f}, 95% CI [{lower_metric:.3f}-{upper_metric:.3f}])' if label_prefix else f'{metric_name} = {mean_metric:.3f} (95% CI)'
    ci_label = '95% CI'

    # 绘图
    ax.plot(base_x, mean_y, label=label, color=color)
    ax.fill_between(base_x, lower_y, upper_y, color=color, alpha=0.2, label=ci_label)

    if curve_type == 'ROC':
        ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Chance')
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
    elif curve_type == 'PR':
        baseline = np.mean(y_true_valid)
        ax.plot([0, 1], [baseline, baseline], linestyle='--', color='grey', label=f'Baseline ({baseline:.2f})')
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.05])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f'Bootstrapped {curve_type} Curve (N={n_bootstrap})')
    ax.legend(loc=loc)
    ax.grid(True)
    plt.tight_layout()
    return ax # 返回轴对象
# ================= 主 K-Fold 函数 =================
def run_kfold_analysis_for_model(model_name, model_template, X_all, y_all, feature_names, X_index, config):
    """执行 K-Fold 分析，包含训练、评估和 SHAP 分析"""
    model_base_dir = os.path.join(config["output_dir_base"], model_name.lower().replace(" ", "_").replace(".", ""))
    os.makedirs(model_base_dir, exist_ok=True)
    logger = setup_logging(model_base_dir, f'{model_name}_kfold_run.log')
    logger.info(f"Starting K-Fold analysis for model: {model_name}")

    n_splits = config["n_splits_cv"]
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config["random_state"])

    fold_metrics = defaultdict(list)
    all_y_true_folds = []
    all_y_proba_folds = []
    all_fold_results_dfs = [] # 新增：用于存储每折的详细结果
    all_shap_values_folds = []
    all_X_test_scaled_folds = []
    all_fprs = []
    all_tprs = []
    all_recalls = []
    all_precisions = []
    all_expected_values = []  # 存储每折的expected_value，用于聚合分析
    aggregated_shap_importance = None # 用于累积SHAP重要性

    needs_scaling = model_name in ['SVM', 'Logistic Regression', 'Neural Networks', 'K-Nearest Neighbors', 'Gaussian Naive Bayes', 'ElasticNet Logistic Regression', 'Linear Discriminant Analysis']

    for fold, (train_index, val_index) in enumerate(kf.split(X_all, y_all)):
        fold_num = fold + 1
        fold_dir = os.path.join(model_base_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)
        # 使用 fold_dir 创建 logger，确保每个 fold 日志独立
        fold_logger = setup_logging(fold_dir, f'fold_{fold_num}_analysis.log')
        fold_logger.info(f"--- Processing Fold {fold_num}/{n_splits} for {model_name} ---")

        X_train_fold, X_val_fold = X_all[train_index], X_all[val_index]
        y_train_fold, y_val_fold = y_all[train_index], y_all[val_index]
        fold_logger.info(f"Train shape: {X_train_fold.shape}, Validation shape: {X_val_fold.shape}")
        fold_logger.info(f"Train label distribution: {np.bincount(y_train_fold)}")
        fold_logger.info(f"Validation label distribution: {np.bincount(y_val_fold)}")


        # 使用 fold logger 记录 DataFrame 创建
        fold_logger.debug("Creating DataFrames for the current fold...")
        X_train_fold_df = pd.DataFrame(X_train_fold, columns=feature_names, index=X_index[train_index])
        X_val_fold_df = pd.DataFrame(X_val_fold, columns=feature_names, index=X_index[val_index])
        fold_logger.debug("DataFrames created.")


        scaler = None
        # 重要：确保后续使用的数据是 scaled DFs
        X_train_fold_for_fit = X_train_fold_df.copy()
        X_val_fold_for_eval = X_val_fold_df.copy()

        if needs_scaling:
            fold_logger.info(f"Applying StandardScaler for {model_name}...")
            scaler = StandardScaler()
            # Fit on training data
            X_train_fold_scaled = scaler.fit_transform(X_train_fold_df)
            # Transform validation data
            X_val_fold_scaled = scaler.transform(X_val_fold_df)
            # Convert back to DataFrame for consistency and feature names
            X_train_fold_for_fit = pd.DataFrame(X_train_fold_scaled, columns=feature_names, index=X_train_fold_df.index)
            X_val_fold_for_eval = pd.DataFrame(X_val_fold_scaled, columns=feature_names, index=X_val_fold_df.index)
            scaler_path = os.path.join(fold_dir, f"scaler_fold_{fold_num}.joblib")
            joblib.dump(scaler, scaler_path)
            fold_logger.info(f"Scaler saved to {scaler_path}")
        else:
             fold_logger.info(f"StandardScaler not required for {model_name}.")


        fold_logger.info("Training model...")
        # 确保每次都创建一个新的模型实例
        if model_name == 'Voting Classifier':
            # For VotingClassifier, the template itself contains fitted estimators, so we re-use it.
            # A better approach would be to re-train the base estimators for each fold, but for simplicity we use the pre-defined one.
            # Note: This is a simplification. A rigorous approach would re-train the base models on each fold's training data.
            current_model = model_template
        else:
            current_model = model_template.__class__(**model_template.get_params())
        try:
            # 使用处理后的数据进行训练
            current_model.fit(X_train_fold_for_fit, y_train_fold)
            fold_logger.info("Model training complete.")
            model_path = os.path.join(fold_dir, f"model_fold_{fold_num}.joblib")
            joblib.dump(current_model, model_path)
            fold_logger.info(f"Model saved to {model_path}")
        except Exception as train_e:
            fold_logger.error(f"Failed to train model for fold {fold_num}: {train_e}", exc_info=True)
            # 记录NaN指标，确保后续聚合不错位
            for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'brier']:
                fold_metrics[key].append(np.nan)
            # 添加占位符或空数据以保持列表长度一致
            all_y_true_folds.append(y_val_fold)
            all_y_proba_folds.append(np.full(y_val_fold.shape, np.nan))
            all_shap_values_folds.append(None)
            all_X_test_scaled_folds.append(None)
            all_fprs.append(np.array([]))
            all_tprs.append(np.array([]))
            all_recalls.append(np.array([]))
            all_precisions.append(np.array([]))
            all_expected_values.append(np.nan) # 添加 NaN expected value
            gc.collect()
            continue # 跳到下一折

        fold_logger.info("Evaluating model on validation set...")
        # 使用处理后的数据进行评估
        y_pred_val = current_model.predict(X_val_fold_for_eval)
        y_pred_proba_val = None

        # 获取概率或得分
        try:
            if hasattr(current_model, 'predict_proba'):
                y_pred_proba_val = current_model.predict_proba(X_val_fold_for_eval)[:, 1]
                fold_logger.debug("Using predict_proba.")
            elif hasattr(current_model, 'decision_function'):
                y_pred_proba_val = current_model.decision_function(X_val_fold_for_eval)
                # 可能需要将 decision_function 的输出归一化到 [0, 1] 才能用于某些指标或绘图
                # 这里暂时不处理，但要注意其范围可能不是概率
                fold_logger.info("Using decision_function output as scores.")
            else:
                fold_logger.warning(f"Model {model_name} does not have predict_proba or decision_function. Cannot calculate ROC/PR AUC.")
                y_pred_proba_val = np.full(y_val_fold.shape, np.nan) # 用 NaN 填充
        except Exception as pred_e:
             fold_logger.error(f"Error getting probabilities/scores: {pred_e}", exc_info=True)
             y_pred_proba_val = np.full(y_val_fold.shape, np.nan)


        # 存储真实标签和预测概率
        all_y_true_folds.append(y_val_fold)
        all_y_proba_folds.append(y_pred_proba_val)

        # 计算指标
        acc = accuracy_score(y_val_fold, y_pred_val)
        prec = precision_score(y_val_fold, y_pred_val, zero_division=0)
        rec = recall_score(y_val_fold, y_pred_val, zero_division=0)
        f1 = f1_score(y_val_fold, y_pred_val, zero_division=0)
        roc_auc_val, pr_auc_val, brier_val = np.nan, np.nan, np.nan

        # 仅在有有效概率且多于一个类别时计算 AUC 和绘图
        if y_pred_proba_val is not None and not np.all(np.isnan(y_pred_proba_val)) and len(np.unique(y_val_fold)) > 1:
            try:
                roc_auc_val = roc_auc_score(y_val_fold, y_pred_proba_val)
                pr_auc_val = average_precision_score(y_val_fold, y_pred_proba_val) # 使用 average_precision_score
                # 添加Brier Score计算
                brier_val = brier_score_loss(y_val_fold, y_pred_proba_val)

                fpr, tpr, _ = roc_curve(y_val_fold, y_pred_proba_val)
                precision_vals, recall_vals, _ = precision_recall_curve(y_val_fold, y_pred_proba_val)

                all_fprs.append(fpr)
                all_tprs.append(tpr)
                all_recalls.append(recall_vals)
                all_precisions.append(precision_vals)

                # 绘制单折曲线
                plot_roc_curve(y_val_fold, y_pred_proba_val, f"{model_name} - Fold {fold_num} ROC", os.path.join(fold_dir, "roc_curve.pdf"), n_bootstrap=500) # 减少 bootstrap 次数加速
                plot_pr_curve(y_val_fold, y_pred_proba_val, f"{model_name} - Fold {fold_num} PR", os.path.join(fold_dir, "pr_curve.pdf"), n_bootstrap=500)
            except Exception as curve_e:
                fold_logger.error(f"Error calculating/plotting ROC/PR curves for fold {fold_num}: {curve_e}", exc_info=True)
                # 添加空数组以保持长度
                all_fprs.append(np.array([]))
                all_tprs.append(np.array([]))
                all_recalls.append(np.array([]))
                all_precisions.append(np.array([]))
        else:
            fold_logger.warning("Skipping ROC/PR curve calculation due to invalid probabilities or single class.")
            all_fprs.append(np.array([]))
            all_tprs.append(np.array([]))
            all_recalls.append(np.array([]))
            all_precisions.append(np.array([]))

        # 存储指标
        fold_metrics['accuracy'].append(acc)
        fold_metrics['precision'].append(prec)
        fold_metrics['recall'].append(rec)
        fold_metrics['f1'].append(f1)
        fold_metrics['roc_auc'].append(roc_auc_val)
        fold_metrics['pr_auc'].append(pr_auc_val)
        fold_metrics['brier'].append(brier_val)  # 添加Brier Score到指标

        # 绘制混淆矩阵
        cm = confusion_matrix(y_val_fold, y_pred_val)
        plot_confusion_matrix(cm, ['Environmental', 'Clinical'], f"{model_name} - Fold {fold_num} CM", os.path.join(fold_dir, "confusion_matrix.pdf"))

        # 新增：保存混淆矩阵中的菌株信息
        val_strain_names = X_index[val_index]
        fold_results_df = pd.DataFrame({
            'Strain': val_strain_names,
            'True_Label': y_val_fold,
            'Predicted_Label': y_pred_val
        })
        all_fold_results_dfs.append(fold_results_df) # 为聚合分析收集数据

        tn_strains = fold_results_df[(fold_results_df['True_Label'] == 0) & (fold_results_df['Predicted_Label'] == 0)]['Strain'].tolist()
        fp_strains = fold_results_df[(fold_results_df['True_Label'] == 0) & (fold_results_df['Predicted_Label'] == 1)]['Strain'].tolist()
        fn_strains = fold_results_df[(fold_results_df['True_Label'] == 1) & (fold_results_df['Predicted_Label'] == 0)]['Strain'].tolist()
        tp_strains = fold_results_df[(fold_results_df['True_Label'] == 1) & (fold_results_df['Predicted_Label'] == 1)]['Strain'].tolist()

        cm_details_df = pd.DataFrame({
            'TN (True_Env)': pd.Series(tn_strains),
            'FP (Pred_Cli)': pd.Series(fp_strains),
            'FN (Pred_Env)': pd.Series(fn_strains),
            'TP (True_Cli)': pd.Series(tp_strains)
        })
        cm_details_path = os.path.join(fold_dir, 'confusion_matrix_details.csv')
        cm_details_df.to_csv(cm_details_path, index=False, encoding='utf-8')
        fold_logger.info(f"Confusion matrix strain details saved to {cm_details_path}")


        # 保存详细指标报告
        report_str = classification_report(y_val_fold, y_pred_val, zero_division=0, target_names=['Environmental', 'Clinical'])
        metrics_path = os.path.join(fold_dir, f'{model_name}_fold_{fold_num}_detailed_metrics.txt')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"===== {model_name} - Fold {fold_num} Detailed Metrics =====\n")
            f.write(f"Samples: Train={len(y_train_fold)}, Validation={len(y_val_fold)}\n\n")
            f.write(f"Accuracy: {acc:.4f}\nPrecision (Clinical): {prec:.4f}\nRecall (Clinical): {rec:.4f}\nF1 Score (Clinical): {f1:.4f}\n")
            f.write(f"ROC AUC: {roc_auc_val:.4f}\nPR AUC (Avg Precision): {pr_auc_val:.4f}\nBrier Score: {brier_val:.4f}\n\n")  # 添加Brier Score输出
            f.write("--- Classification Report ---\n")
            f.write(report_str)
            f.write("\n\n--- Confusion Matrix ---\n")
            f.write(f"                 Predicted\n")
            f.write(f"True           Environmental | Clinical\n")
            f.write(f"Environmental      {cm[0, 0]:<10} | {cm[0, 1]:<10}\n")
            f.write(f"Clinical           {cm[1, 0]:<10} | {cm[1, 1]:<10}\n")
        fold_logger.info(f"Fold metrics saved to {metrics_path}")

        # --- SHAP 分析 ---
        fold_shap_values_2d = None
        interaction_values_3d = None
        expected_value_fold = np.nan # 初始化为 NaN

        if model_name in config["shap"]["shap_analysis_models"]:
            fold_logger.info("Starting SHAP analysis for this fold...")
            try:
                n_bkg = min(config["shap"]["background_samples"], X_train_fold_for_fit.shape[0])
                if n_bkg < 1:
                    fold_logger.warning(f"Background sample count {n_bkg} too small, skipping SHAP.")
                else:
                    # 使用训练集（拟合模型的数据）作为背景数据
                    background_data = shap.sample(X_train_fold_for_fit, n_bkg, random_state=config["random_state"])
                    fold_logger.info(f"Using {background_data.shape[0]} samples from training set as SHAP background.")


                    explainer = None
                    # *** 扩展的SHAP Explainer逻辑 ***
                    tree_models = (XGBClassifier, RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier)
                    if LGBM_AVAILABLE:
                        tree_models += (LGBMClassifier,)

                    if isinstance(current_model, tree_models):
                        explainer = shap.TreeExplainer(
                            current_model,
                            background_data,
                            feature_names=feature_names,
                            feature_perturbation="interventional"
                        )
                        fold_logger.info(f"Using TreeExplainer with interventional perturbation for {type(current_model).__name__}")
                    elif isinstance(current_model, LogisticRegression):
                        explainer = shap.LinearExplainer(
                            current_model,
                            background_data,
                            feature_names=feature_names
                        )
                        fold_logger.info("Using LinearExplainer for Logistic Regression")
                    elif isinstance(current_model, (SVC, MLPClassifier, KNeighborsClassifier, GaussianNB, AdaBoostClassifier, LinearDiscriminantAnalysis)):
                        if hasattr(current_model, 'predict_proba'):
                            predict_proba_func = lambda x: current_model.predict_proba(x)
                            explainer = shap.KernelExplainer(predict_proba_func, background_data, feature_names=feature_names, link="logit")
                            fold_logger.info(f"Using KernelExplainer for {type(current_model).__name__}")
                        else:
                            fold_logger.warning(f"Cannot use KernelExplainer for {model_name} as it lacks a 'predict_proba' method.")


                    if explainer:
                        fold_logger.info("Calculating SHAP values on validation set...")
                        shap_values_fold = explainer.shap_values(X_val_fold_for_eval)
                        fold_logger.info("SHAP values calculated.")
                        # 获取 expected_value
                        if hasattr(explainer, 'expected_value'):
                            expected_value_fold = explainer.expected_value
                            fold_logger.debug(f"Raw expected value: {expected_value_fold}")
                        else:
                            fold_logger.warning("Explainer does not have expected_value attribute.")
                            if hasattr(shap_values_fold, 'base_values'):
                                expected_value_fold = shap_values_fold.base_values
                                fold_logger.info("Using base_values from shap_values object.")
                        # 处理多输出 (列表) vs 单输出 (数组)
                        if isinstance(shap_values_fold, list) and len(shap_values_fold) > 1:
                            # 列表输出时通常为 [class0_shap, class1_shap]
                            fold_shap_values_2d = np.array(shap_values_fold[1]) if len(shap_values_fold) > 1 else np.array(shap_values_fold[0])
                            # 对应 expected_value 也取 class=1
                            if isinstance(expected_value_fold, (list, np.ndarray)) and len(expected_value_fold) > 1:
                                expected_value_fold = expected_value_fold[1]
                            fold_logger.debug(f"Extracted SHAP list values for class 1: {fold_shap_values_2d.shape}")
                        elif isinstance(shap_values_fold, np.ndarray):
                            if shap_values_fold.ndim == 3:
                                # (samples, features, classes)
                                fold_shap_values_2d = shap_values_fold[:, :, 1]
                                # 对应 expected_value 也取 class=1
                                if isinstance(expected_value_fold, (list, np.ndarray)) and len(expected_value_fold) > 1:
                                    expected_value_fold = expected_value_fold[1]
                                fold_logger.debug(f"Extracted SHAP 3D values for class 1: {fold_shap_values_2d.shape}")
                            elif shap_values_fold.ndim == 2:
                                # (samples, features)
                                fold_shap_values_2d = shap_values_fold
                                fold_logger.debug(f"Using 2D SHAP values directly: {fold_shap_values_2d.shape}")
                            else:
                                fold_shap_values_2d = None
                        else:
                            fold_shap_values_2d = None
                        if fold_shap_values_2d is None:
                            fold_logger.warning(f"Fold {fold_num}: 无法提取 2D SHAP 值 (raw shape: {getattr(shap_values_fold, 'shape', 'N/A')}), 跳过 SHAP 绘图。")
                        # 处理多输出 (列表) vs 单输出 (数组)
                        if isinstance(expected_value_fold, (list, np.ndarray)) and len(expected_value_fold) > 1:
                            expected_value_fold = expected_value_fold[1]
                        fold_logger.info(f"Final expected value for fold: {expected_value_fold}")
                        all_expected_values.append(expected_value_fold)
                        
                         # --- SHAP Interaction Values (仅对 TreeExplainer) ---
                        interaction_values_3d = None # 初始化
                        if fold_shap_values_2d is not None and isinstance(explainer, shap.TreeExplainer):
                             fold_logger.info("Calculating SHAP interaction values...")
                             try:
                                 # 使用验证集的一个子集计算交互值，减少计算量
                                 n_interact_samples = min(config["shap"]["heatmap_sample_size"], X_val_fold_for_eval.shape[0])
                                 X_val_interact_sample_df = shap.sample(X_val_fold_for_eval, n_interact_samples, random_state=config["random_state"])
                                 fold_logger.debug(f"Calculating interactions on {X_val_interact_sample_df.shape[0]} samples.")

                                 interaction_values_fold = explainer.shap_interaction_values(X_val_interact_sample_df)
                                 fold_logger.info("SHAP interaction values calculated.")

                                 # 处理交互值输出格式
                                 if isinstance(interaction_values_fold, list) and len(interaction_values_fold) > 1:
                                     interaction_values_3d = interaction_values_fold[1] # 取类别 1 的交互值
                                     fold_logger.debug(f"Extracted 3D interaction values for class 1. Shape: {interaction_values_3d.shape}")
                                 elif isinstance(interaction_values_fold, np.ndarray) and interaction_values_fold.ndim == 3:
                                     interaction_values_3d = interaction_values_fold
                                     fold_logger.debug(f"Using 3D interaction values directly. Shape: {interaction_values_3d.shape}")
                                 else:
                                     fold_logger.warning(f"Unexpected SHAP interaction values format: type {type(interaction_values_fold)}. Cannot plot interactions.")
                                     interaction_values_3d = None
                             except Exception as interact_e:
                                  fold_logger.error(f"Error calculating SHAP interaction values: {interact_e}", exc_info=True)
                                  interaction_values_3d = None


                        # --- SHAP 绘图和数据保存 ---
                        if fold_shap_values_2d is not None and not np.isnan(expected_value_fold):
                            # 添加到聚合列表
                            all_shap_values_folds.append(fold_shap_values_2d)
                            all_X_test_scaled_folds.append(X_val_fold_for_eval) # 存储用于绘图的验证集数据

                            # 创建 Explanation 对象
                            explanation_fold = shap.Explanation(
                                values=fold_shap_values_2d,
                                base_values=np.full(fold_shap_values_2d.shape[0], expected_value_fold), # 确保 base_values 形状匹配
                                data=X_val_fold_for_eval,
                                feature_names=feature_names
                            )
                            fold_logger.info(f"Created SHAP Explanation object for fold {fold_num}.")


                            # 1. 特征重要性 (基于 SHAP)
                            mean_abs_shap = np.abs(fold_shap_values_2d).mean(axis=0)
                            shap_importance_series = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
                            # 累加到聚合重要性
                            if aggregated_shap_importance is None:
                                aggregated_shap_importance = pd.Series(np.zeros(len(feature_names)), index=feature_names)
                            aggregated_shap_importance = aggregated_shap_importance.add(shap_importance_series, fill_value=0)

                            plot_feature_importance(shap_importance_series, f"{model_name} Fold {fold_num} SHAP Importance", os.path.join(fold_dir, "shap_feature_importance.pdf"))

                            # 保存 Top N 特征列表
                            for n in config["top_n_features"]:
                                top_n_genes = shap_importance_series.head(min(n, len(shap_importance_series))).index.tolist()
                                fold_logger.info(f"Fold {fold_num} Top {n} Features (SHAP): {top_n_genes}")
                                top_n_filename = os.path.join(fold_dir, f"top_{n}_features_shap.txt")
                                try:
                                    with open(top_n_filename, 'w', encoding='utf-8') as f_top:
                                        for gene in top_n_genes:
                                            f_top.write(str(gene) + '\n') # 确保写入字符串
                                    fold_logger.info(f"Top {n} features saved to {top_n_filename}")
                                except Exception as write_e:
                                     fold_logger.error(f"Failed to write top features file {top_n_filename}: {write_e}")


                            # 2. SHAP Summary Plot
                            plot_title_prefix = f"{model_name} Fold {fold_num}"
                            plot_shap_summary(fold_shap_values_2d, X_val_fold_for_eval, feature_names, 
                                      f"{plot_title_prefix} SHAP Summary (Dot)",
                                      os.path.join(fold_dir, "shap_summary_dot_plot.pdf"), plot_type="dot")

                            # 3. SHAP Heatmap Plot
                            sample_size_heatmap = min(config["shap"]["heatmap_sample_size"], explanation_fold.shape[0])
                            fold_logger.info(f"Generating SHAP heatmap for first {sample_size_heatmap} samples...")
                            # 确保传递 Explanation 对象的子集
                            plot_shap_heatmap_fold(
                                explanation_fold[:sample_size_heatmap],
                                f"{plot_title_prefix} SHAP Heatmap",
                                os.path.join(fold_dir, "shap_heatmap.pdf")
                            )

                            # 4. SHAP Decision Plot
                            plot_shap_decision_fold(explanation_fold, config["shap"]["decision_plot_n_samples"], f"{plot_title_prefix} SHAP Decision Plot", os.path.join(fold_dir, "shap_decision_plot.pdf"))

                            # 5. SHAP Interaction (Dependence) Plots
                            if interaction_values_3d is not None:
                                fold_logger.info("Processing SHAP interaction plots...")
                                # 计算平均绝对交互值以找出 Top 交互对
                                mean_abs_interaction = np.abs(interaction_values_3d).mean(axis=0)
                                if mean_abs_interaction.ndim == 2 and mean_abs_interaction.shape[0] == mean_abs_interaction.shape[1]:
                                    np.fill_diagonal(mean_abs_interaction, -np.inf) # 忽略自身交互，设为负无穷方便排序忽略
                                    flat_indices = mean_abs_interaction.flatten().argsort()[::-1] # 降序索引
                                    top_interaction_features_fold = []
                                    seen_pairs = set()
                                    n_feat_int = mean_abs_interaction.shape[0]

                                    for flat_idx in flat_indices:
                                        idx1, idx2 = np.unravel_index(flat_idx, mean_abs_interaction.shape)
                                        if idx1 < len(feature_names) and idx2 < len(feature_names):
                                            f1, f2 = feature_names[idx1], feature_names[idx2]
                                            sp = tuple(sorted((f1, f2))) # 创建排序后的元组以避免重复 (A,B) 和 (B,A)
                                            if f1 != f2 and sp not in seen_pairs: # 确保不是自身交互且未被记录
                                                top_interaction_features_fold.append((f1, f2))
                                                seen_pairs.add(sp)
                                                if len(top_interaction_features_fold) >= config["shap"]["top_n_interaction_features"]:
                                                    break # 达到所需数量
                                    fold_logger.info(f"Top {len(top_interaction_features_fold)} interaction pairs identified: {top_interaction_features_fold}")

                                    # 确保传递给绘图函数的数据维度匹配
                                    if n_interact_samples == interaction_values_3d.shape[0]:
                                         shap_values_interact_sample = fold_shap_values_2d[:n_interact_samples, :]

                                         plot_shap_interactions_fold(
                                             interaction_values_3d,
                                             X_val_interact_sample_df, # 传递用于计算交互的 X 子集
                                             shap_values_interact_sample, # 传递对应的 SHAP 值子集
                                             feature_names,
                                             top_interaction_features_fold,
                                             plot_title_prefix,
                                             os.path.join(fold_dir, f"shap_interaction") # 文件名前缀
                                         )
                                    else:
                                         fold_logger.warning("Interaction plot skipped: Sample size mismatch between interaction values and original SHAP values.")

                                else:
                                    fold_logger.warning("Could not determine top interaction pairs from interaction values.")

                        else:
                            fold_logger.warning(f"Skipping SHAP plots for fold {fold_num} due to missing SHAP values or expected value.")
                            # 添加 None 以保持列表长度一致
                            all_shap_values_folds.append(None)
                            all_X_test_scaled_folds.append(None)
                            all_expected_values.append(np.nan) # 记录 NaN

            except Exception as shap_e:
                fold_logger.error(f"Error during SHAP analysis for fold {fold_num}: {shap_e}", exc_info=True)
                # 确保即使出错也添加占位符
                all_shap_values_folds.append(None)
                all_X_test_scaled_folds.append(None)
                # 检查 all_expected_values 长度是否已增加，如果未增加则添加 NaN
                if len(all_expected_values) < fold_num:
                    all_expected_values.append(np.nan)

        fold_logger.info(f"--- Finished Processing Fold {fold_num} ---")
        # 清理内存
        del X_train_fold, X_val_fold, y_train_fold, y_val_fold
        del X_train_fold_df, X_val_fold_df
        del X_train_fold_for_fit, X_val_fold_for_eval
        if 'current_model' in locals(): del current_model
        if 'explainer' in locals(): del explainer
        if 'shap_values_fold' in locals(): del shap_values_fold
        if 'fold_shap_values_2d' in locals(): del fold_shap_values_2d
        if 'interaction_values_fold' in locals(): del interaction_values_fold
        if 'explanation_fold' in locals(): del explanation_fold
        gc.collect()

    logger.info(f"--- Completed {n_splits} folds for model: {model_name} ---")

    # --- 聚合分析 ---
    agg_dir = os.path.join(model_base_dir, "aggregated")
    os.makedirs(agg_dir, exist_ok=True)
    agg_logger = setup_logging(agg_dir, 'aggregated_analysis.log') # 使用聚合目录的 logger
    agg_logger.info(f"--- Starting Aggregated Analysis for {model_name} ---")

    # 1. 聚合指标
    mean_metrics, std_metrics = {}, {}
    agg_logger.info("Aggregated Metrics across folds:")
    metrics_str_list = []
    for metric, values in fold_metrics.items():
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            mean_metrics[metric] = np.mean(valid_values)
            std_metrics[metric] = np.std(valid_values)
            metric_str = f"  Mean {metric.upper():<10}: {mean_metrics[metric]:.4f} (+/- {std_metrics[metric]:.4f})"
            agg_logger.info(metric_str)
            metrics_str_list.append(metric_str)
        else:
            mean_metrics[metric] = np.nan
            std_metrics[metric] = np.nan
            agg_logger.info(f"  Mean {metric.upper()}: NaN (no valid data)")

    # 保存聚合指标到 CSV
    metrics_summary_df = pd.DataFrame({
        'Metric': list(mean_metrics.keys()),
        'Mean': list(mean_metrics.values()),
        'StdDev': list(std_metrics.values())
    })
    metrics_summary_path = os.path.join(agg_dir, f"{model_name}_aggregated_metrics.csv")
    metrics_summary_df.round(4).to_csv(metrics_summary_path, index=False)
    agg_logger.info(f"Aggregated metrics saved to {metrics_summary_path}")

    # 新增：聚合混淆矩阵分析
    if all_fold_results_dfs:
        aggregated_results_df = pd.concat(all_fold_results_dfs, ignore_index=True)
        
        # 计算并绘制聚合混淆矩阵
        y_true_agg_pred = aggregated_results_df['True_Label']
        y_pred_agg = aggregated_results_df['Predicted_Label']
        agg_cm = confusion_matrix(y_true_agg_pred, y_pred_agg)
        plot_confusion_matrix(agg_cm, ['Environmental', 'Clinical'], f"{model_name} - Aggregated CM", os.path.join(agg_dir, "aggregated_confusion_matrix.pdf"))
        agg_logger.info("Aggregated confusion matrix plotted.")

        # 获取并保存聚合混淆矩阵的菌株详情
        agg_tn_strains = aggregated_results_df[(aggregated_results_df['True_Label'] == 0) & (aggregated_results_df['Predicted_Label'] == 0)]['Strain'].tolist()
        agg_fp_strains = aggregated_results_df[(aggregated_results_df['True_Label'] == 0) & (aggregated_results_df['Predicted_Label'] == 1)]['Strain'].tolist()
        agg_fn_strains = aggregated_results_df[(aggregated_results_df['True_Label'] == 1) & (aggregated_results_df['Predicted_Label'] == 0)]['Strain'].tolist()
        agg_tp_strains = aggregated_results_df[(aggregated_results_df['True_Label'] == 1) & (aggregated_results_df['Predicted_Label'] == 1)]['Strain'].tolist()

        agg_cm_details_df = pd.DataFrame({
            'TN (True_Env)': pd.Series(agg_tn_strains),
            'FP (Pred_Cli)': pd.Series(agg_fp_strains),
            'FN (Pred_Env)': pd.Series(agg_fn_strains),
            'TP (True_Cli)': pd.Series(agg_tp_strains)
        })
        agg_cm_details_path = os.path.join(agg_dir, 'aggregated_confusion_matrix_details.csv')
        agg_cm_details_df.to_csv(agg_cm_details_path, index=False, encoding='utf-8')
        agg_logger.info(f"Aggregated confusion matrix strain details saved to {agg_cm_details_path}")
    else:
        agg_logger.warning("No fold results available to generate aggregated confusion matrix.")


    # 2. Bootstrapped 聚合曲线
    y_true_agg = np.concatenate(all_y_true_folds) if all_y_true_folds else np.array([])
    # Filter out folds where y_proba might be None or all NaNs before concatenating
    valid_proba_folds = [p for p in all_y_proba_folds if p is not None and not np.all(np.isnan(p))]
    y_proba_agg = np.concatenate(valid_proba_folds) if valid_proba_folds else np.array([])


    if len(y_true_agg) > 0 and len(y_proba_agg) > 0 and len(np.unique(y_true_agg)) > 1:
        agg_logger.info(f"Plotting bootstrapped curves using {len(y_true_agg)} aggregated samples.")
        # Bootstrapped ROC
        fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
        plot_bootstrapped_curve(y_true_agg, y_proba_agg, curve_type='ROC',
                                n_bootstrap=config["bootstrap_ci"]["n_iterations"],
                                title=f'{model_name} Aggregated Bootstrapped ROC Curve',
                                color='darkorange', label_prefix=model_name, ax=ax_roc)
        fig_roc.savefig(os.path.join(agg_dir, "bootstrapped_agg_roc_curve.pdf"))
        plt.close(fig_roc)

        # Bootstrapped PR
        fig_pr, ax_pr = plt.subplots(figsize=(8, 8))
        plot_bootstrapped_curve(y_true_agg, y_proba_agg, curve_type='PR',
                                n_bootstrap=config["bootstrap_ci"]["n_iterations"],
                                title=f'{model_name} Aggregated Bootstrapped PR Curve',
                                color='blue', label_prefix=model_name, ax=ax_pr)
        fig_pr.savefig(os.path.join(agg_dir, "bootstrapped_agg_pr_curve.pdf"))
        plt.close(fig_pr)
        agg_logger.info("Bootstrapped aggregated ROC and PR curves plotted.")
    else:
        agg_logger.warning("Not enough aggregated data (or valid probabilities) to plot bootstrapped curves.")


    # 3. CV 均值/标准差曲线
    if any(len(f) > 0 for f in all_fprs) and any(len(t) > 0 for t in all_tprs):
        agg_logger.info("Plotting CV mean/std ROC curve.")
        mean_fpr = np.linspace(0, 1, 100)
        interp_tprs = []
        valid_fold_roc_aucs = [fold_metrics['roc_auc'][i] for i in range(n_splits) if not np.isnan(fold_metrics['roc_auc'][i])]

        for i, (fpr, tpr) in enumerate(zip(all_fprs, all_tprs)):
             if len(fpr) > 1 and len(tpr) > 1: # Need at least 2 points for interpolation
                 try:
                     interp_tpr = np.interp(mean_fpr, fpr, tpr)
                     interp_tpr[0] = 0.0
                     interp_tprs.append(interp_tpr)
                 except Exception as interp_e:
                      agg_logger.warning(f"Could not interpolate TPR for fold {i+1}: {interp_e}")

        if interp_tprs:
            mean_tpr = np.mean(interp_tprs, axis=0)
            std_tpr = np.std(interp_tprs, axis=0)
            roc_auc_mean = np.mean(valid_fold_roc_aucs) if valid_fold_roc_aucs else np.nan
            roc_auc_std = np.std(valid_fold_roc_aucs) if valid_fold_roc_aucs else np.nan

            plt.figure(figsize=(8, 6))
            plt.plot(mean_fpr, mean_tpr, color='darkorange',
                    label=f'Mean ROC (AUC = {roc_auc_mean:.3f} ± {roc_auc_std:.3f})')
            tprs_upper = mean_tpr + std_tpr
            tprs_lower = mean_tpr - std_tpr
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey',
                           alpha=0.2, label=f'±1 std. dev.')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} CV Mean ROC Curve (Based on {len(interp_tprs)} Folds)')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(agg_dir, "cv_roc_curve_mean_std.pdf"))
            plt.close()
            agg_logger.info("CV ROC curve mean and std plotted.")
        else:
             agg_logger.warning("No valid interpolated TPRs found for CV ROC plot.")


        if any(len(p) > 0 for p in all_precisions) and any(len(r) > 0 for r in all_recalls):
            agg_logger.info("Plotting CV mean/std PR curve.")
            mean_recall_pr = np.linspace(0, 1, 100)
            interp_precisions = []
            valid_fold_pr_aucs = [fold_metrics['pr_auc'][i] for i in range(n_splits) if not np.isnan(fold_metrics['pr_auc'][i])]
            y_true_baseline = np.mean(y_true_agg) if len(y_true_agg) > 0 else 0.5 # Use aggregated baseline

            for i, (precision, recall) in enumerate(zip(all_precisions, all_recalls)):
                if len(precision) > 1 and len(recall) > 1:
                    try:
                        # Sort recall
                        sorted_idx = np.argsort(recall)
                        recall_sorted = recall[sorted_idx]
                        precision_sorted = precision[sorted_idx]
                        # Ensure 0 and 1 recall points
                        if recall_sorted[0] > 0:
                            recall_sorted = np.concatenate([[0], recall_sorted])
                            precision_sorted = np.concatenate([[precision_sorted[0]], precision_sorted])
                        if recall_sorted[-1] < 1:
                            recall_sorted = np.concatenate([recall_sorted, [1]])
                            precision_sorted = np.concatenate([precision_sorted, [y_true_baseline]]) # Use overall baseline

                        interp_precision = np.interp(mean_recall_pr, recall_sorted, precision_sorted)
                        interp_precisions.append(interp_precision)
                    except Exception as interp_e:
                        agg_logger.warning(f"Could not interpolate Precision for fold {i+1}: {interp_e}")


            if interp_precisions:
                mean_precision = np.mean(interp_precisions, axis=0)
                std_precision = np.std(interp_precisions, axis=0)
                pr_auc_mean = np.mean(valid_fold_pr_aucs) if valid_fold_pr_aucs else np.nan
                pr_auc_std = np.std(valid_fold_pr_aucs) if valid_fold_pr_aucs else np.nan

                plt.figure(figsize=(8, 6))
                plt.plot(mean_recall_pr, mean_precision, color='blue',
                        label=f'Mean PR (AP = {pr_auc_mean:.3f} ± {pr_auc_std:.3f})')
                precisions_upper = mean_precision + std_precision
                precisions_lower = mean_precision - std_precision
                plt.fill_between(mean_recall_pr, precisions_lower, precisions_upper,
                               color='grey', alpha=0.2, label=f'±1 std. dev.')

                plt.plot([0, 1], [y_true_baseline, y_true_baseline], 'k--', label=f'Baseline ({y_true_baseline:.2f})')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'{model_name} CV Mean PR Curve (Based on {len(interp_precisions)} Folds)')
                plt.legend(loc="lower left")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(agg_dir, "cv_pr_curve_mean_std.pdf"))
                plt.close()
                agg_logger.info("CV PR curve mean and std plotted.")
            else:
                 agg_logger.warning("No valid interpolated Precisions found for CV PR plot.")

        else:
            agg_logger.warning("Not enough per-fold curve data to plot CV mean/std curves.")

    else:
        agg_logger.warning("Not enough aggregated data (or valid probabilities) to plot bootstrapped curves.")


    # 4. 聚合 SHAP 分析
    valid_shap_folds = [sv for sv in all_shap_values_folds if sv is not None and isinstance(sv, np.ndarray) and sv.ndim == 2]
    valid_X_test_scaled_folds = [Xdf for Xdf in all_X_test_scaled_folds if Xdf is not None and isinstance(Xdf, pd.DataFrame) and Xdf.shape[0] > 0]
    # 处理 all_expected_values：将可能的数组转换为标量并过滤 None/NaN
    valid_expected_values = []
    for ev in all_expected_values:
        if ev is None:
            continue
        # 将可能的数组转换为标量
        try:
            if isinstance(ev, np.ndarray):
                ev_scalar = float(np.nanmean(ev))
            else:
                ev_scalar = float(ev)
        except Exception:
            continue
        # 过滤 NaN
        if not np.isnan(ev_scalar):
            valid_expected_values.append(ev_scalar)

    agg_logger.info(f"Found {len(valid_shap_folds)} valid SHAP value sets, {len(valid_X_test_scaled_folds)} valid X_test sets, and {len(valid_expected_values)} valid expected values.")


    # 检查模型是否在 SHAP 分析列表中，并且有有效数据
    if model_name not in config["shap"]["shap_analysis_models"]:
        agg_logger.info(f"Skipping aggregated SHAP for {model_name} as it's not in the analysis list.")
    elif not valid_shap_folds or not valid_X_test_scaled_folds or len(valid_shap_folds) != len(valid_X_test_scaled_folds):
        agg_logger.warning(f"Skipping aggregated SHAP for {model_name} due to missing or mismatched SHAP/X_test data across folds.")
    else:
        agg_logger.info(f"Aggregating SHAP values from {len(valid_shap_folds)} folds...")
        try:
            # 检查特征数量是否一致
            num_features = valid_shap_folds[0].shape[1]
            if not all(sv.shape[1] == num_features for sv in valid_shap_folds) or not all(Xdf.shape[1] == num_features for Xdf in valid_X_test_scaled_folds):
                agg_logger.error(f"Inconsistent feature dimensions across folds. Cannot aggregate SHAP.")
            else:
                # 堆叠 SHAP 值和特征数据
                all_shap_values_stacked = np.vstack(valid_shap_folds)
                all_X_test_scaled_stacked_df = pd.concat(valid_X_test_scaled_folds, axis=0, ignore_index=True)

                agg_logger.info(f"Stacked SHAP values shape: {all_shap_values_stacked.shape}")
                agg_logger.info(f"Stacked X_test shape: {all_X_test_scaled_stacked_df.shape}")


                if all_shap_values_stacked.shape[0] != all_X_test_scaled_stacked_df.shape[0]:
                    agg_logger.error(f"Stacked SHAP samples ({all_shap_values_stacked.shape[0]}) mismatch with X_test samples ({all_X_test_scaled_stacked_df.shape[0]}). Aborting aggregated SHAP.")
                else:
                    # --- MODIFICATION START ---
                    agg_logger.info("Calculating SHAP importance and robust contribution direction...")
                    
                    # 1. 计算平均绝对SHAP值，用于最终排序
                    mean_abs_shap_agg = np.abs(all_shap_values_stacked).mean(axis=0)
                    
                    # 2. 根据您提出的更优逻辑计算贡献方向
                    contribution_directions = []
                    for i, feature in enumerate(feature_names):
                        # a. 找到所有该特征存在的样本 (特征值为1)
                        feature_present_mask = all_X_test_scaled_stacked_df.iloc[:, i] == 1
                        # c. 找到所有该特征缺失的样本 (特征值为0)
                        feature_absent_mask = all_X_test_scaled_stacked_df.iloc[:, i] == 0

                        # b. 计算特征存在时的SHAP值的平均值
                        mean_shap_when_present = 0
                        if feature_present_mask.any():
                            mean_shap_when_present = all_shap_values_stacked[feature_present_mask, i].mean()
                        
                        # d. 计算特征缺失时的SHAP值的平均值
                        mean_shap_when_absent = 0
                        if feature_absent_mask.any():
                            mean_shap_when_absent = all_shap_values_stacked[feature_absent_mask, i].mean()

                        # e. 计算净效应 (存在的影响 - 缺失的影响) 来判断方向
                        net_effect = mean_shap_when_present - mean_shap_when_absent

                        if net_effect > 0:
                            contribution_directions.append('Positive (Clinical)')
                        elif net_effect < 0:
                            contribution_directions.append('Negative (Environmental)')
                        else:
                            contribution_directions.append('Neutral')

                    # 3. 创建最终的DataFrame
                    agg_shap_importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'MeanAbsoluteShap': mean_abs_shap_agg,
                        'ContributionDirection': contribution_directions
                    })

                    # 4. 仍然根据MeanAbsoluteShap进行排序，以确定最重要的特征
                    agg_shap_importance_df = agg_shap_importance_df.sort_values(by='MeanAbsoluteShap', ascending=False)
                    
                    agg_logger.info("Calculated SHAP importance with robust contribution direction based on net effect.")
                    # --- MODIFICATION END ---

                    agg_importance_path = os.path.join(agg_dir, "aggregated_shap_feature_importance.csv")
                    agg_shap_importance_df.round(6).to_csv(agg_importance_path, index=False)
                    agg_logger.info(f"Aggregated SHAP importance with direction saved to {agg_importance_path}")

                    # =================================================================
                    # 修正 1: 统一特征排序
                    # 从新保存的CSV文件中读取全局特征排序，作为所有后续绘图的唯一标准
                    # =================================================================
                    try:
                        # 使用新的DataFrame来获取排序
                        globally_ordered_feature_names = agg_shap_importance_df['Feature'].tolist()
                        agg_logger.info(f"Successfully loaded globally ordered features from calculated importance. Using this order for all plots.")

                        # 根据全局顺序，重新排列聚合的 SHAP 值和特征数据
                        original_feature_list = all_X_test_scaled_stacked_df.columns.tolist()
                        
                        if not all(f in original_feature_list for f in globally_ordered_feature_names):
                            agg_logger.error("Feature mismatch between ordered list and stacked data. Cannot reorder.")
                            globally_ordered_feature_names = original_feature_list
                            all_shap_values_stacked_ordered = all_shap_values_stacked
                            all_X_test_scaled_stacked_df_ordered = all_X_test_scaled_stacked_df
                        else:
                            sorter = np.array([original_feature_list.index(f) for f in globally_ordered_feature_names])
                            all_shap_values_stacked_ordered = all_shap_values_stacked[:, sorter]
                            all_X_test_scaled_stacked_df_ordered = all_X_test_scaled_stacked_df[globally_ordered_feature_names]
                            agg_logger.info("Reordered stacked SHAP values and feature data according to global importance.")

                    except Exception as e_reorder:
                        agg_logger.error(f"Could not apply global feature order. Plots may have inconsistent ordering. Error: {e_reorder}")
                        globally_ordered_feature_names = feature_names
                        all_shap_values_stacked_ordered = all_shap_values_stacked
                        all_X_test_scaled_stacked_df_ordered = all_X_test_scaled_stacked_df

                    # 保存 Top N 特征 (基于绝对值排序)
                    for n in config["top_n_features"]:
                        top_n_agg = agg_shap_importance_df['Feature'].head(min(n, len(agg_shap_importance_df))).tolist()
                        agg_logger.info(f"Aggregated Top {n} Features (SHAP): {top_n_agg}")
                        agg_top_n_filename = os.path.join(agg_dir, f"top_{n}_features_agg_shap.txt")
                        try:
                            with open(agg_top_n_filename, 'w', encoding='utf-8') as f_top:
                                for gene in top_n_agg:
                                    f_top.write(str(gene) + '\n')
                            agg_logger.info(f"Aggregated top {n} features saved to {agg_top_n_filename}")
                        except Exception as write_e:
                            agg_logger.error(f"Failed to write aggregated top features file {agg_top_n_filename}: {write_e}")

                    # 绘制聚合特征重要性条形图 (使用绝对值排序的Series)
                    bar_plot_series = pd.Series(agg_shap_importance_df['MeanAbsoluteShap'].values, index=agg_shap_importance_df['Feature'])
                    plot_feature_importance(bar_plot_series, f"{model_name} Aggregated SHAP Importance",
                                            os.path.join(agg_dir, "aggregated_shap_feature_importance_bar.pdf"))

                    # --- 聚合 SHAP 图 (使用排序后的数据) ---
                    n_total_samples = all_shap_values_stacked_ordered.shape[0]
                    n_plot_samples = min(config["shap"]["max_shap_samples_agg"], n_total_samples)
                    if n_plot_samples < n_total_samples:
                        agg_logger.info(f"Sampling {n_plot_samples} out of {n_total_samples} for aggregated SHAP plots.")
                        plot_indices = np.random.choice(n_total_samples, n_plot_samples, replace=False)
                        X_plot_agg = all_X_test_scaled_stacked_df_ordered.iloc[plot_indices]
                        shap_values_plot_agg = all_shap_values_stacked_ordered[plot_indices, :]
                    else:
                        X_plot_agg = all_X_test_scaled_stacked_df_ordered
                        shap_values_plot_agg = all_shap_values_stacked_ordered

                    mean_base_value_agg = np.mean(valid_expected_values) if valid_expected_values else 0
                    agg_logger.info(f"Using mean expected value for aggregated plots: {mean_base_value_agg:.4f} (from {len(valid_expected_values)} folds)")

                    # 1. 聚合 SHAP Summary Plot (使用排序后的数据)
                    agg_logger.info(f"Plotting aggregated SHAP summary (dot) with {X_plot_agg.shape[0]} samples.")
                    plot_shap_summary(shap_values_plot_agg, X_plot_agg, globally_ordered_feature_names, 
                                      f"{model_name} Aggregated SHAP Summary (Sampled from Folds)",
                                      os.path.join(agg_dir, "aggregated_shap_summary_dot_plot.pdf"), plot_type="dot")

                    # 2. 聚合热图使用所有样本
                    agg_logger.info(f"Generating aggregated SHAP heatmap with ALL {n_total_samples} samples.")
                    agg_explanation_heatmap = shap.Explanation(
                        values=all_shap_values_stacked_ordered,
                        base_values=np.full(n_total_samples, mean_base_value_agg),
                        data=all_X_test_scaled_stacked_df_ordered,
                        feature_names=globally_ordered_feature_names
                    )
                    plot_shap_heatmap_fold(
                        agg_explanation_heatmap,
                        f"{model_name} Aggregated SHAP Heatmap (All Samples)",
                        os.path.join(agg_dir, "aggregated_shap_heatmap.pdf")
                    )

                    # 3. 聚合 SHAP Decision Plot (使用采样和排序后的数据)
                    n_decision_samples_agg = min(config["shap"]["decision_plot_n_samples"], X_plot_agg.shape[0])
                    agg_logger.info(f"Generating aggregated SHAP decision plot with {n_decision_samples_agg} samples")
                    # 创建聚合 Explanation 对象 (使用子集)
                    agg_explanation_decision = shap.Explanation(
                        values=shap_values_plot_agg[:n_decision_samples_agg],
                        base_values=np.full(n_decision_samples_agg, mean_base_value_agg),
                        data=X_plot_agg.iloc[:n_decision_samples_agg],
                        feature_names=globally_ordered_feature_names
                    )
                    plot_shap_decision_fold(
                        agg_explanation_decision,
                        n_decision_samples_agg, # Pass the actual number of samples
                        f"{model_name} Aggregated SHAP Decision Plot",
                        os.path.join(agg_dir, "aggregated_shap_decision_plot.pdf")
                    )

                    if model_name in ['XGBoost', 'Random Forest', 'Gradient Boosting', 'LightGBM', 'Decision Tree']:
                        agg_logger.info("Aggregated SHAP interaction analysis would require aggregating interaction matrices (not implemented here).")
                        pass

        except Exception as agg_shap_e:
            agg_logger.error(f"Error during aggregated SHAP analysis: {agg_shap_e}", exc_info=True)
            agg_logger.error(traceback.format_exc())

    agg_logger.info(f"--- Finished Aggregated Analysis for {model_name} ---")

    results_summary = {f"mean_{metric}": mean_metrics[metric] for metric in mean_metrics}
    results_summary.update({f"std_{metric}": std_metrics.get(metric, np.nan) for metric in mean_metrics})
    gc.collect()
    return results_summary

# ================= 主程序 =================
if __name__ == "__main__":
    main_output_dir = CONFIG["output_dir_base"]
    os.makedirs(main_output_dir, exist_ok=True)
    main_logger = setup_logging(main_output_dir, 'main_program_run.log')
    main_logger.info("="*30 + " Program Start " + "="*30)
    main_logger.info(f"Time: {pd.Timestamp.now()}")
    main_logger.info(f"Output Base Directory: {main_output_dir}")
    main_logger.info(f"K-Fold Splits: {CONFIG['n_splits_cv']}")
    main_logger.info(f"CPU Count: {os.cpu_count()}")
    main_logger.info(f"Bootstrapping Enabled: {CONFIG['bootstrap_ci']['enabled']} (N={CONFIG['bootstrap_ci']['n_iterations']})")
    main_logger.info(f"SHAP Analysis Models: {CONFIG['shap']['shap_analysis_models']}")

    try:
        main_logger.info("Step 1: Loading Data...")
        merged_df = load_data(main_logger)

        main_logger.info("Step 2: Preprocessing Full Data...")
        X_all_np, y_all_np, feature_names, data_index = preprocess_data_full(merged_df, main_logger)
        if X_all_np is None or y_all_np is None or len(feature_names) == 0:
            raise ValueError("完整数据集预处理失败或无有效特征，程序终止。")
        main_logger.info(f"Data ready: X shape {X_all_np.shape}, y shape {y_all_np.shape}, Features: {len(feature_names)}")
        # 释放原始 DataFrame 内存
        del merged_df
        gc.collect()

        main_logger.info("Step 3: Defining Models...")
        # 定义模型模板时，考虑并行性设置，n_jobs=-1 使用所有核心，但可能导致内存问题，n_jobs=1 是单核
        # 可以根据机器资源调整 n_jobs
        N_JOBS = max(1, os.cpu_count() // 2) # 使用一半核心，或至少1个
        main_logger.info(f"Setting n_jobs={N_JOBS} for applicable models.")
        all_models_templates = {
            'XGBoost': XGBClassifier(random_state=CONFIG["random_state"], use_label_encoder=False, eval_metric='logloss', n_jobs=N_JOBS),
            'Random Forest': RandomForestClassifier(random_state=CONFIG["random_state"], n_jobs=N_JOBS, n_estimators=200, min_samples_leaf=5, class_weight='balanced'),
            'SVM': SVC(probability=True, random_state=CONFIG["random_state"], cache_size=500, kernel='rbf', C=1.0, gamma='scale', class_weight='balanced'),
            'Logistic Regression': LogisticRegression(random_state=CONFIG["random_state"], max_iter=5000, solver='liblinear', C=1.0, class_weight='balanced'),
            'ElasticNet Logistic Regression': LogisticRegression(random_state=CONFIG["random_state"], max_iter=5000, solver='saga', penalty='elasticnet', l1_ratio=0.5, class_weight='balanced'),
            'Neural Networks': MLPClassifier(random_state=CONFIG["random_state"], max_iter=1000, hidden_layer_sizes=(100, 50), activation='relu', solver='adam', early_stopping=True, n_iter_no_change=20, alpha=0.0001, learning_rate='adaptive'),
            'Gradient Boosting': GradientBoostingClassifier(random_state=CONFIG["random_state"], n_estimators=200, min_samples_leaf=5),
            'AdaBoost': AdaBoostClassifier(random_state=CONFIG["random_state"], n_estimators=200),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS),
            'Gaussian Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=CONFIG["random_state"], min_samples_leaf=5, class_weight='balanced'),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis()
        }
        if LGBM_AVAILABLE:
            all_models_templates['LightGBM'] = LGBMClassifier(random_state=CONFIG["random_state"], n_jobs=N_JOBS, n_estimators=200, class_weight='balanced')
            main_logger.info("LightGBM is available and added to the model list.")
        else:
            main_logger.info("LightGBM not found. Skipping LightGBM model. Please install with 'pip install lightgbm'")

        main_logger.info(f"Models to be trained: {list(all_models_templates.keys())}")


        main_logger.info("Step 4: Running K-Fold Analysis for Individual Models...")
        overall_results_summary = {}
        for model_name, model_template in all_models_templates.items():
            main_logger.info("\n" + "=" * 20 + f" Processing Model: {model_name} " + "=" * 20 + "\n")
            try:
                model_agg_metrics = run_kfold_analysis_for_model(
                    model_name, model_template,
                    X_all_np, y_all_np, # 传递 NumPy 数组
                    feature_names,      # 传递特征名列表
                    data_index,         # 传递原始索引
                    CONFIG
                )
                overall_results_summary[model_name] = model_agg_metrics
                main_logger.info(f"Finished processing {model_name}.")
            except Exception as model_run_e:
                main_logger.error(f"Error running analysis for {model_name}: {model_run_e}", exc_info=True)
                overall_results_summary[model_name] = {"error": str(model_run_e)}
            # 清理内存
            gc.collect()

        main_logger.info("="*30 + " All Individual Models Processed " + "="*30)

        # --- Voting Classifier Section ---
        main_logger.info("\n" + "=" * 20 + " Processing Voting Classifier " + "=" * 20 + "\n")
        try:
            # 从结果中筛选出成功的模型并排序
            successful_models = {
                name: metrics for name, metrics in overall_results_summary.items()
                if isinstance(metrics, dict) and 'error' not in metrics and not np.isnan(metrics.get('mean_roc_auc', np.nan))
            }
            
            if len(successful_models) >= 2:
                sorted_models = sorted(successful_models.items(), key=lambda item: item[1]['mean_roc_auc'], reverse=True)
                
                # 选择Top 5模型，如果不足5个则全选
                top_n_for_voting = min(5, len(sorted_models))
                top_models_for_voting = sorted_models[:top_n_for_voting]
                
                main_logger.info(f"Selected Top {top_n_for_voting} models for Voting Classifier based on ROC AUC:")
                for name, metrics in top_models_for_voting:
                    main_logger.info(f"  - {name} (Mean ROC AUC: {metrics['mean_roc_auc']:.4f})")

                # 创建投票分类器的估计器列表
                # 注意：这里我们使用模型模板，这意味着它们是未训练的。这在K-fold循环中是正确的，因为它们会在每个fold中被重新训练。
                estimators = [(name, all_models_templates[name]) for name, _ in top_models_for_voting]
                
                # 创建并评估投票分类器
                voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=N_JOBS)
                
                voting_results = run_kfold_analysis_for_model(
                    'Voting Classifier',
                    voting_clf,
                    X_all_np, y_all_np,
                    feature_names, data_index,
                    CONFIG
                )
                overall_results_summary['Voting Classifier'] = voting_results
                main_logger.info("Finished processing Voting Classifier.")
            else:
                main_logger.info("Not enough successful models (at least 2 required) to build a Voting Classifier. Skipping.")

        except Exception as voting_e:
            main_logger.error(f"Error during Voting Classifier analysis: {voting_e}", exc_info=True)
            overall_results_summary['Voting Classifier'] = {"error": str(voting_e)}
        gc.collect()

        # --- Final Summary ---
        main_logger.info("="*30 + " Final Summary Generation " + "="*30)
        summary_file = os.path.join(main_output_dir, "all_models_aggregated_summary.csv")
        summary_list = []
        # 定义列顺序
        metric_keys = ['roc_auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'f1', 'brier']  # 添加brier
        cols_order = ['Model', 'Status'] + [f'Mean {key.upper()}' for key in metric_keys] + [f'Std {key.upper()}' for key in metric_keys]

        for model_name, metrics_dict in overall_results_summary.items():
            row = {'Model': model_name}
            if isinstance(metrics_dict, dict) and 'error' in metrics_dict:
                row['Status'] = 'Error'
                # 添加 NaN 值以保持列完整
                for key in metric_keys:
                    row[f'Mean {key.upper()}'] = np.nan
                    row[f'Std {key.upper()}'] = np.nan
            elif isinstance(metrics_dict, dict):
                row['Status'] = 'Completed'
                for key in metric_keys:
                    row[f'Mean {key.upper()}'] = metrics_dict.get(f'mean_{key}', np.nan)
                    row[f'Std {key.upper()}'] = metrics_dict.get(f'std_{key}', np.nan)
            else: # 处理意外情况
                 row['Status'] = 'Unknown'
                 for key in metric_keys:
                    row[f'Mean {key.upper()}'] = np.nan
                    row[f'Std {key.upper()}'] = np.nan

            summary_list.append(row)

        if summary_list:
            summary_df = pd.DataFrame(summary_list)
            # 确保所有预期的列都存在，即使全是 NaN
            for col in cols_order:
                if col not in summary_df.columns:
                    summary_df[col] = np.nan
            # 重新排序
            summary_df = summary_df[cols_order]
            # 保存
            summary_df.to_csv(summary_file, index=False, float_format='%.4f')
            main_logger.info(f"Overall summary saved to {summary_file}")
            main_logger.info("\n--- Aggregated Performance Summary ---\n" + summary_df.to_string(index=False, float_format='%.4f') + "\n------------------------------------")
        else:
             main_logger.warning("No summary data generated.")


    except Exception as e_main:
        main_logger.error(f"主程序发生意外错误: {str(e_main)}", exc_info=True)
        main_logger.error(traceback.format_exc()) # 打印完整堆栈跟踪
    finally:
        main_logger.info("="*30 + " Program End " + "="*30)
        main_logger.info(f"\n==================== Run Summary ====================")
        main_logger.info(f"主输出目录: {main_output_dir}")
        main_logger.info("查看各模型子目录（例如 'xgboost', 'voting_classifier'）获取详细结果：")
        main_logger.info("  - `aggregated/` 包含交叉验证的平均指标、 bootstrapped 和 CV 曲线图和 aggregated SHAP 图。")
        main_logger.info("  - `fold_N/` 包含第 N 折训练的模型、评估图表、SHAP 分析和 Top 特征列表。")
        main_logger.info(f"主日志文件: {os.path.join(main_output_dir, 'main_program_run.log')}")
        main_logger.info("=====================================================")
        logging.shutdown() # 关闭所有日志处理器
        gc.collect()
