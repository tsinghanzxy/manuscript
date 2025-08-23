# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import joblib
import warnings

# 忽略不必要的警告，使输出更整洁
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ======================= 用户配置 - 修改为全局配置 =======================
GLOBAL_CONFIG = {
    # 1. 输入的新数据文件路径 (对所有模型类型都一样)
    "input_excel_path": "48LP_pangenome_mapping_results_80cov_90_iden_modified.xlsx",

    # 2. 预测结果的最终输出文件路径 (所有模型及所有fold结果都将写入此文件)
    "output_excel_path": "all_4147LP_48LP_all_models_and_folds_integrated_prediction_results.xlsx", # 新的输出文件名

    # 3. 原始训练结果的主目录 (假设所有模型都在同一个主目录下)
    "fixed_model_base_dir": "all_4147LP_kfold_results", # 如果你的模型都放在这个目录下

    # 4. 指定要循环使用的所有算法模型类型
    "all_model_types": ['xgboost', 'random_forest', 'svm', 'logistic_regression', 'neural_networks'],

    # 5. 包含训练时所用全部特征的txt文件路径
    "fixed_features_file_name": "training_feature_names.txt", # 如果特征文件名称固定在每个模型基目录下

    # 6. K-fold 的折数 (必须与训练时一致)
    "n_splits": 5,

    # 7. 预测为 'Clinical' 的概率阈值 (此版本代码将不再使用此阈值进行最终分类)
    "prediction_threshold": 0.5
}
# ==========================================================


def load_features(filepath):
    """从文件加载训练时使用的特征列表"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            features = [line.strip() for line in f]
        print(f"成功加载 {len(features)} 个训练特征，来源: {filepath}")
        return features
    except FileNotFoundError:
        print(f"错误：找不到特征文件 '{filepath}'。")
        print("请确保您已通过修改并运行原始训练脚本生成了 'training_feature_names.txt' 文件。")
        raise


def load_and_preprocess_new_data(filepath, training_features):
    """加载并预处理新的泛基因组数据以进行预测"""
    print(f"正在加载新数据: {filepath}")
    try:
        # 脚本假设基因是行，菌株是列，因此需要转置 (.T)
        # index_col=0 将第一列（基因名）作为索引
        new_data_df = pd.read_excel(filepath, index_col=0).T
        print(f"数据加载成功，原始形状: {new_data_df.shape} (菌株数, 基因数)")
    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{filepath}'。")
        raise
    except Exception as e:
        print(f"加载新数据时出错: {e}")
        raise

    # 核心步骤：将新数据的列与训练特征对齐
    # .reindex() 会确保新数据DataFrame的列与 training_features 完全一致
    # 多的列会被丢弃，少的列（即新数据中没有的训练特征）会被添加并用0填充
    print("正在将新数据特征与训练特征对齐...")
    processed_df = new_data_df.reindex(columns=training_features, fill_value=0)

    # 检查是否有任何列在对齐后全是NaN（这不应该发生，因为我们用0填充）
    if processed_df.isnull().values.any():
        print("警告：处理后的数据中发现NaN值，将用0填充。")
        processed_df.fillna(0, inplace=True)

    print(f"数据预处理完成，最终形状: {processed_df.shape}")
    return processed_df, processed_df.index.tolist()


def get_all_probabilities_for_model_type(current_model_type,
                                         model_base_dir,
                                         input_data_df):
    """
    针对单一模型类型执行 K-fold 集成预测，并返回每个fold的Clinical概率列表
    和平均Clinical概率。
    """
    print("=" * 30)
    print(f"开始执行模型类型 '{current_model_type.upper()}' 的集成预测...")
    print("=" * 30)

    all_probas_clinical_folds = [] # 存储每个fold的 Clinical 概率数组

    # 构建当前模型类型的具体文件夹路径
    model_type_folder = os.path.join(model_base_dir, current_model_type)

    # 检查模型文件夹是否存在
    if not os.path.isdir(model_type_folder):
        print(f"错误：找不到模型文件夹 '{model_type_folder}'。请确保该类型模型已训练并存放正确。")
        return None, None # 返回None表示该模型类型未成功处理

    print("\n正在加载模型并进行预测...")
    for i in range(1, GLOBAL_CONFIG['n_splits'] + 1):
        fold_dir = os.path.join(model_type_folder, f"fold_{i}")
        model_path = os.path.join(fold_dir, f"model_fold_{i}.joblib")

        try:
            # 加载模型
            model = joblib.load(model_path)
            print(f"  - 已加载模型: {model_path}")

            # 准备当前fold的数据副本
            fold_data = input_data_df.copy()

            # 如果模型需要缩放 (SVM, LR, NN)，则加载并应用对应的scaler
            needs_scaling = current_model_type in ['svm', 'logistic_regression', 'neural_networks']
            if needs_scaling:
                scaler_path = os.path.join(fold_dir, f"scaler_fold_{i}.joblib")
                if os.path.exists(scaler_path): # 确保scaler文件存在
                    scaler = joblib.load(scaler_path)
                    fold_data_scaled = scaler.transform(fold_data)
                    # 将缩放后的numpy数组转回DataFrame，保持列名和索引
                    fold_data = pd.DataFrame(fold_data_scaled, index=fold_data.index, columns=fold_data.columns)
                    print(f"    - 已应用数据缩放: {scaler_path}")
                else:
                    print(f"    - 警告：模型类型 '{current_model_type}' 通常需要scaler，但未找到文件 '{scaler_path}'。跳过缩放。")


            # 使用模型预测概率
            # predict_proba返回一个数组，[[P(class 0), P(class 1)], ...]
            # class 0 通常是 'Environmental'，class 1 是 'Clinical'
            probas = model.predict_proba(fold_data)
            all_probas_clinical_folds.append(probas[:, 1]) # Clinical 概率 for current fold

        except FileNotFoundError:
            print(f"错误：在 '{fold_dir}' 中找不到模型或scaler文件。跳过 fold {i}。")
            continue
        except Exception as e:
            print(f"处理 fold {i} 时出错: {e}")
            continue

    if not all_probas_clinical_folds:
        print(f"\n警告：未能从模型类型 '{current_model_type}' 的任何fold成功加载模型并进行预测。")
        return None, None

    # 聚合预测结果（计算平均值）
    print("\n正在聚合所有模型的预测结果并计算平均值...")
    mean_probas_clinical = np.mean(np.stack(all_probas_clinical_folds, axis=0), axis=0)
    print(f"模型类型 '{current_model_type.upper()}' 预测完成。")
    print("=" * 30)
    return all_probas_clinical_folds, mean_probas_clinical


if __name__ == "__main__":
    # 在循环外部只加载一次特征文件和新数据，避免重复加载
    try:
        # 确定特征文件路径，假设特征文件在 GLOBAL_CONFIG['fixed_model_base_dir'] 下
        fixed_base_dir = GLOBAL_CONFIG['fixed_model_base_dir']
        features_path_for_all_models = os.path.join(fixed_base_dir, GLOBAL_CONFIG['fixed_features_file_name'])

        training_features = load_features(features_path_for_all_models)
        input_data_df, strain_names = load_and_preprocess_new_data(
            GLOBAL_CONFIG['input_excel_path'],
            training_features
        )
    except Exception as e:
        print(f"初始化加载数据或特征时出错，程序终止: {e}")
        exit() # 退出程序

    if input_data_df.empty:
        print("没有可供预测的数据。程序终止。")
        exit()

    # 初始化最终的结果DataFrame
    final_results_df = pd.DataFrame({'Strain_Name': strain_names})

    # 循环遍历所有模型类型并进行预测，然后将结果添加到 final_results_df
    for model_type in GLOBAL_CONFIG['all_model_types']:
        # 获取当前模型类型的所有fold概率和平均概率
        all_folds_probas, mean_probas = get_all_probabilities_for_model_type(
            current_model_type=model_type,
            model_base_dir=GLOBAL_CONFIG['fixed_model_base_dir'],
            input_data_df=input_data_df
        )

        if all_folds_probas is not None and mean_probas is not None:
            # 添加每个fold的概率列
            for i in range(GLOBAL_CONFIG['n_splits']):
                # 规范化列名，例如：XGBoost_Fold_1_Clinical_Probability
                col_name_fold = f'{model_type.replace(" ", "_")}_Fold_{i+1}_Clinical_Probability'
                final_results_df[col_name_fold] = all_folds_probas[i]

            # 添加平均概率列
            # 规范化列名，例如：XGBoost_Average_Clinical_Probability
            col_name_avg = f'{model_type.replace(" ", "_")}_Average_Clinical_Probability'
            final_results_df[col_name_avg] = mean_probas
        else:
            print(f"警告：模型类型 '{model_type}' 的预测未能成功，将不会包含在最终结果中。")

    # 对所有概率列进行格式化 (在保存前转换为百分比字符串)
    # 遍历所有列名，找到包含 'Probability' 字符串的列进行格式化
    for col in final_results_df.columns:
        if 'Probability' in col:
            final_results_df[col] = final_results_df[col].map('{:.2%}'.format)

    # 最终保存整合后的Excel文件
    try:
        final_results_df.to_excel(GLOBAL_CONFIG['output_excel_path'], index=False)
        print("\n所有模型类型及所有fold的预测已整合完成！")
        print(f"结果已成功保存至: {GLOBAL_CONFIG['output_excel_path']}")
        print("="*30)
    except Exception as e:
        print(f"\n错误：保存最终结果文件失败: {e}")