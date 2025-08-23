import os
import time
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor
from Bio import SeqIO
from Bio.Seq import Seq
from queue import Queue

# 配置参数
input_dir = "4147_LP_fna_part1"
output_dir = "dignal_PCR_results"
output_file = os.path.join(output_dir, "pcr_results_sg1_newprimer_part1.xlsx")
forward_primer = "TGCAGCAAGCAAAAGTTCAG"
reverse_primer = "AATAAGGGTGAATACAAAGTACATC"
max_mismatch = 6
min_tm = 40.0
min_product_length = 300
max_product_length = 400

# 真阳性/真阴性文件列表
true_positives = ["GZBYS23.3(sg1).fna", "SZGMW65.25(sg1).fna"]
true_negatives = ["GZYXW351(non-sg1).fna", "SZGMS2-10(non-sg1).fna"]

# 性能优化配置
MAX_WORKERS = 32  # 根据CPU核心数调整
SHOW_DETAILED_PROGRESS = True  # 是否显示每个序列的详细处理过程

# 线程安全变量
result_queue = Queue()
processed_files = 0
amplified_files = 0
start_time = time.time()
lock = threading.Lock()

def print_header():
    """显示程序启动信息"""
    print(f"""
==============================================
 In silico PCR 分析程序（多线程优化版）
----------------------------------------------
输入目录: {input_dir}
输出文件: {output_file}
线程数量: {MAX_WORKERS}
参数设置:
   - 正向引物: {forward_primer} (Tm: {calculate_tm(forward_primer):.1f}°C)
   - 反向引物: {reverse_primer} (Tm: {calculate_tm(reverse_primer):.1f}°C)
   - 最大错配数: {max_mismatch}
   - 产物长度范围: {min_product_length}-{max_product_length} bp
   - 最低退火温度: {min_tm}°C
==============================================
    """)

def print_footer():
    """显示运行摘要"""
    duration = time.time() - start_time
    print(f"""
==============================================
 分析完成！
----------------------------------------------
 处理文件总数: {processed_files}
 成功扩增文件数: {amplified_files}
 总耗时: {duration:.1f} 秒
 平均处理速度: {duration/processed_files:.1f} 秒/文件
==============================================
    """)

def calculate_tm(primer):
    """计算引物退火温度（简化版Wallace规则）"""
    return 4 * (primer.count('G') + primer.count('C')) + 2 * (len(primer) - (primer.count('G') + primer.count('C')))

def find_matches(template, primer, max_mismatch):
    """在模板中寻找引物结合位点（允许错配）"""
    matches = []
    primer_len = len(primer)
    for i in range(len(template) - primer_len + 1):
        substr = template[i:i+primer_len]
        mismatch = sum(1 for a, b in zip(substr, primer) if a != b)
        if mismatch <= max_mismatch:
            matches.append((i, i + primer_len, mismatch))
    return matches

def analyze_strand(seq_record, forward_primer, reverse_primer, strand):
    """分析指定链方向的产物"""
    if strand == "reverse":
        template = str(seq_record.seq.reverse_complement()).upper()
    else:
        template = str(seq_record.seq).upper()
    
    forward_matches = find_matches(template, forward_primer, max_mismatch)
    rev_comp_reverse = str(Seq(reverse_primer).reverse_complement())
    reverse_matches = find_matches(template, rev_comp_reverse, max_mismatch)
    
    products = []
    for f_start, f_end, f_miss in forward_matches:
        for r_start, r_end, r_miss in reverse_matches:
            if f_end <= r_start:  # 确保产物方向正确
                product_start = f_start
                product_end = r_end
                product_length = product_end - product_start
                
                if min_product_length <= product_length <= max_product_length:
                    product_seq = template[product_start:product_end]
                    tm_f = calculate_tm(forward_primer)
                    tm_r = calculate_tm(reverse_primer)
                    
                    if tm_f >= min_tm and tm_r >= min_tm:
                        products.append({
                            "strand": strand,
                            "start": product_start if strand == "forward" else len(template) - product_end,
                            "end": product_end if strand == "forward" else len(template) - product_start,
                            "length": product_length,
                            "sequence": product_seq,
                            "mismatch": f_miss + r_miss,
                            "tm_f": tm_f,
                            "tm_r": tm_r
                        })
    return products

def find_pcr_products(seq_record):
    """同时分析正向和反向链"""
    forward_products = analyze_strand(seq_record, forward_primer, reverse_primer, "forward")
    reverse_products = analyze_strand(seq_record, forward_primer, reverse_primer, "reverse")
    
    all_products = forward_products + reverse_products
    all_products.sort(key=lambda x: (x['mismatch'], -x['tm_f']))
    return all_products

def process_file(filename):
    """处理单个文件"""
    global processed_files, amplified_files
    
    try:
        # 读取文件内容
        with open(os.path.join(input_dir, filename), "r") as f:
            records = list(SeqIO.parse(f, "fasta"))
        
        file_results = []
        for record in records:
            if SHOW_DETAILED_PROGRESS:
                with lock:
                    print(f"  ▷ 分析序列 {record.id[:20]}... ({len(record.seq)} bp)")
            
            products = find_pcr_products(record)
            if products:
                best_product = products[0]
                file_results.append({
                    "File": filename,
                    "Strand": best_product["strand"],
                    "Start Position": best_product["start"],
                    "End Position": best_product["end"],
                    "Product Length": best_product["length"],
                    "Product Sequence": best_product["sequence"],
                    "Total Mismatches": best_product["mismatch"],
                    "Tm Forward (°C)": best_product["tm_f"],
                    "Tm Reverse (°C)": best_product["tm_r"],
                    "Amplification": "Yes"
                })
        
        # 更新统计信息
        with lock:
            processed_files += 1
            if file_results:
                amplified_files += 1
            result_queue.put(file_results if file_results else [{
                "File": filename,
                "Amplification": "No",
                "Strand": "",
                "Start Position": "",
                "End Position": "",
                "Product Length": 0,
                "Product Sequence": "",
                "Total Mismatches": "",
                "Tm Forward (°C)": "",
                "Tm Reverse (°C)": ""
            }])
            
            if SHOW_DETAILED_PROGRESS:
                print(f"  ✔ 文件处理完成: {filename}")
                if file_results:
                    print(f"    ✓ 发现产物！长度: {best_product['length']} bp, 错配: {best_product['mismatch']}")
                    print(f"     Tm: {best_product['tm_f']:.1f}°C/{best_product['tm_r']:.1f}°C")

    except Exception as e:
        with lock:
            print(f"\n文件处理错误: {filename} - {str(e)}")

def main():
    print_header()
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件列表
    file_list = [f for f in os.listdir(input_dir) if f.endswith(".fna")]
    total_files = len(file_list)
    
    # 使用线程池处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(process_file, file_list))
    
    # 收集结果
    results = []
    while not result_queue.empty():
        results.extend(result_queue.get())
    
    # 生成DataFrame
    df = pd.DataFrame(results)
    
    # 性能评估
    tp = len(df[(df['File'].isin(true_positives)) & (df['Amplification'] == "Yes")])
    fp = len(df[(df['File'].isin(true_negatives)) & (df['Amplification'] == "Yes")])
    tn = len(true_negatives) - fp
    fn = len(true_positives) - tp
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 保存结果
    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer, sheet_name="PCR Results", index=False)
        pd.DataFrame({
            "Metric": ["Sensitivity", "Specificity"],
            "Value": [sensitivity, specificity]
        }).to_excel(writer, sheet_name="Performance Metrics", index=False)
    
    print_footer()

if __name__ == "__main__":
    main()