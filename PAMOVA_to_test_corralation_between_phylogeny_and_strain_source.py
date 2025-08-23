import pandas as pd
from Bio import Phylo
from ete3 import Tree, TreeStyle, NodeStyle, AttrFace, TextFace, CircleFace
import matplotlib.pyplot as plt
import io
import numpy as np
from skbio.stats.distance import mantel
import openpyxl
import multiprocessing
import concurrent.futures

# --- 1. 定义文件路径和列名 ---
metadata_file_path = '4147LP_metadata.txt'
core_tree_file_path = 'core.nwk' # 请确保此文件存在
pan_tree_file_path = 'pan.nwk'   # 请确保此文件存在
output_excel_path = 'PAMOVA_Results.xlsx' # 输出Excel文件路径

strain_id_column = 'Strain' # 元数据文件中菌株名的列名 (第一列)
source_column = 'Source'   # 元数据文件中来源的列名 (第四列)

# --- 2. 读取和预处理元数据 (这部分不需要并行化，因为是文件I/O和数据准备) ---
print("--- 正在读取和预处理元数据 ---")
try:
    # 修正: 使用制表符 '\t' 作为分隔符
    metadata = pd.read_csv(metadata_file_path, sep='\t', engine='python')
    print(f"成功读取元数据文件: {metadata_file_path}")
    print("\n元数据预览:")
    print(metadata.head())
    print(f"\n原始元数据中独特的菌株来源类别: {metadata[source_column].unique()}")
    print(f"检测到的元数据列名: {metadata.columns.tolist()}")


    # 筛选掉 'N/A' 的来源，只保留 'Clinical' 和 'Environmental'
    metadata_filtered = metadata[metadata[source_column].isin(['Clinical', 'Environmental'])].copy()
    print(f"\n筛选后保留 {len(metadata_filtered)} 条记录 (只包含 Clinical 和 Environmental 来源)。")
    print(f"筛选后独特的菌株来源类别: {metadata_filtered[source_column].unique()}")

    # 创建一个字典，将菌株ID映射到来源信息
    strain_source_map = dict(zip(metadata_filtered[strain_id_column], metadata_filtered[source_column]))

except FileNotFoundError as e:
    print(f"错误: 文件未找到。请检查文件路径是否正确。{e}")
    exit()
except pd.errors.ParserError as e: # Catch specific pandas parsing errors
    print(f"读取或处理元数据时发生解析错误: {e}")
    print("请检查您的元数据文件 (.txt) 的格式，特别是列数是否一致，以及分隔符是否为制表符。")
    print("如果仍有错误，可能是因为某些字段中包含制表符但未被引号包裹，或文件编码问题。")
    exit()
except Exception as e:
    print(f"读取或处理元数据时发生未知错误: {e}")
    exit()

# --- 3. 定义 PAMOVA (Mantel Test) 函数 ---
# 这个函数将独立运行在不同的进程中
def perform_mantel_test_task(tree_file, strain_source_map, tree_type):
    print(f"\n--- 正在对 {tree_type} 进化树进行 Mantel Test 分析 ---")
    results = {
        'Tree Type': tree_type,
        'Correlation Coefficient (r)': None,
        'P-value': None,
        'Permutations': None,
        'Conclusion': 'Error during analysis'
    }
    
    try:
        # 读取进化树
        tree = Phylo.read(tree_file, "newick")
        print(f"成功读取 {tree_type} 进化树文件: {tree_file}")

        # 获取在元数据和树中都存在的菌株ID
        tree_tip_names = [clade.name for clade in tree.get_terminals()]
        
        temp_common_strains = []
        clade_objects = {} # 用于存储 Bio.Phylo clade 对象
        for strain in sorted([s for s in tree_tip_names if s in strain_source_map]):
            found_clades = tree.find_clades(strain)
            if found_clades:
                clade_obj = next(found_clades, None) # Use None as default if not found
                if clade_obj:
                    clade_objects[strain] = clade_obj
                    temp_common_strains.append(strain)
        
        common_strains = temp_common_strains # 更新 common_strains 列表

        if len(common_strains) < 2:
            print(f"警告: {tree_type} 树和元数据中共同菌株少于2个。跳过 Mantel Test。")
            results['Conclusion'] = 'Less than 2 common strains for distance calculation.'
            return results

        print(f"在 {tree_type} 树和元数据中找到 {len(common_strains)} 个共同菌株进行 Mantel Test。")

        # 1. 计算系统发育距离矩阵
        phylo_dist_matrix = np.zeros((len(common_strains), len(common_strains)))

        for i in range(len(common_strains)):
            clade_i = clade_objects[common_strains[i]]
            for j in range(len(common_strains)):
                clade_j = clade_objects[common_strains[j]]
                phylo_dist_matrix[i, j] = tree.distance(clade_i, clade_j)

        # 2. 构建来源相似性矩阵
        source_dissimilarity_matrix = np.zeros((len(common_strains), len(common_strains)))
        for i in range(len(common_strains)):
            for j in range(len(common_strains)):
                if strain_source_map[common_strains[i]] != strain_source_map[common_strains[j]]:
                    source_dissimilarity_matrix[i, j] = 1

        # 执行 Mantel test
        permutations_count = 9999
        result = mantel(phylo_dist_matrix, source_dissimilarity_matrix, permutations=permutations_count, alternative='two-sided')

        results['Correlation Coefficient (r)'] = f"{result[0]:.4f}"
        results['P-value'] = f"{result[1]:.4f}"
        results['Permutations'] = result[2]

        if result[1] < 0.05:
            results['Conclusion'] = f"显著关联 (P < 0.05)。相同来源的菌株倾向于在进化树上聚类。"
            print(f"结论: 在 {tree_type} 进化树中，菌株的系统发育距离与来源（Clinical vs. Environmental）之间存在显著关联 (P < 0.05)。这意味着相同来源的菌株倾向于在进化树上聚类。")
        else:
            results['Conclusion'] = f"无显著关联 (P >= 0.05)。菌株来源可能没有在进化树上形成明显的聚类。"
            print(f"结论: 在 {tree_type} 进化树中，菌株的系统发育距离与来源（Clinical vs. Environmental）之间没有检测到显著关联 (P >= 0.05)。这意味着菌株来源可能没有在进化树上形成明显的聚类。")

        return results

    except Exception as e:
        print(f"对 {tree_type} 进化树执行 Mantel Test 时发生错误: {e}")
        results['Conclusion'] = f'Error during analysis: {e}'
        return results

# --- 4. 可视化函数 (这个函数在主进程中执行，因为它涉及GUI/文件写入) ---
def visualize_tree_with_sources(tree_path, strain_source_map, tree_title, output_filename_base):
    print(f"\n--- 正在可视化 {tree_title} ---")
    try:
        ete_tree = Tree(tree_path, format=1) # format=1 适用于 Newick 格式

        ts = TreeStyle()
        ts.show_leaf_name = True
        ts.mode = "r" # "c" for circular, "r" for rectangular
        ts.arc_start = -180
        ts.arc_span = 360 # 圆形树完整的360度
        ts.scale = 100
        ts.branch_vertical_margin = 0.5 # 调整分支间距以防止重叠，尤其是对于大树，可根据树的大小调整

        # 定义来源颜色
        source_colors = {
            'Clinical': 'red',
            'Environmental': 'blue',
        }

        # 为每个叶子节点应用样式和添加来源标记
        for leaf in ete_tree.get_leaves():
            node_style = NodeStyle()
            # 默认为灰色，如果找不到匹配的来源
            node_style["fgcolor"] = "gray"
            node_style["size"] = 5
            leaf.add_face(CircleFace(3, "gray", style='circle'), column=0, position="branch-right")


            if leaf.name in strain_source_map: # 检查菌株名是否在过滤后的元数据映射中
                source = strain_source_map[leaf.name]
                if source in source_colors:
                    color = source_colors[source]
                    node_style["fgcolor"] = color
                    node_style["size"] = 10 # 节点大小
                    # 移除默认的灰色圆圈，添加彩色圆圈
                    if leaf.faces: # Clear existing faces if any from default
                        leaf.faces = []
                    leaf.add_face(CircleFace(5, color, style='circle'), column=0, position="branch-right")

            leaf.set_style(node_style)

        # 添加图例
        ts.legend.add_face(TextFace("Legend", fsize=12, fgcolor="black"), column=0)
        for source, color in source_colors.items():
            ts.legend.add_face(CircleFace(5, color, style='circle'), column=0)
            ts.legend.add_face(TextFace(f" {source}", fsize=10), column=1)

        # 修正: 将输出格式更改为 PDF
        output_filename = f"{output_filename_base}.pdf"
        ete_tree.render(output_filename, w=1000, h=1000, tree_style=ts)
        print(f" {tree_title} 树图已保存为 {output_filename}")

    except ImportError:
        print("ETE3 库未安装。请运行 'pip install ete3' 安装。")
    except Exception as e:
        print(f"可视化 {tree_title} 时发生错误: {e}")

# --- 主程序执行 ---
if __name__ == '__main__': # 必须在 if __name__ == '__main__': 块内运行多进程代码
    all_results = []

    # 使用 ProcessPoolExecutor 来并行执行 Mantel Test
    # max_workers=2 因为我们只有两个独立的Mantel Test任务
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        # 提交两个 Mantel Test 任务到进程池
        future_core = executor.submit(perform_mantel_test_task, core_tree_file_path, strain_source_map, "核心基因组 (Core-genome)")
        future_pan = executor.submit(perform_mantel_test_task, pan_tree_file_path, strain_source_map, "泛基因组 (Pan-genome)")

        # 获取任务结果
        core_results = future_core.result()
        pan_results = future_pan.result()

        all_results.append(core_results)
        all_results.append(pan_results)

    print("\n" + "="*50 + "\n") # 分隔线
    print("Mantel Test 计算任务已完成，正在生成可视化图和 Excel 报告。")

    # 可视化部分仍然在主进程中执行
    # 由于树对象不能直接从子进程返回，所以在这里重新加载树文件进行可视化
    visualize_tree_with_sources(core_tree_file_path, strain_source_map, "核心基因组进化树 - 菌株来源", "core_genome_tree_with_sources") # 传递不带扩展名的文件名
    visualize_tree_with_sources(pan_tree_file_path, strain_source_map, "泛基因组进化树 - 菌株来源", "pan_genome_tree_with_sources")   # 传递不带扩展名的文件名


    # --- 将结果输出到 Excel ---
    print(f"\n--- 正在将结果写入 Excel 文件: {output_excel_path} ---")
    try:
        results_df = pd.DataFrame(all_results)
        results_df.to_excel(output_excel_path, index=False)
        print(f"PAMOVA 结果已成功写入 {output_excel_path}")
    except Exception as e:
        print(f"写入 Excel 文件时发生错误: {e}")

    print("\n所有分析完成。")