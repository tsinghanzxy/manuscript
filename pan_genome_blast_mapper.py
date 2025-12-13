import os
import subprocess
import pandas as pd
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys
import shutil

# --- 参数配置 ---
PANGENOME_FASTA = "4147_11734_pan_genome_reference.fa"
ANNOTATION_DIR = "20250608_18_LP_isolates_annontion" ##组装好的质量完善的基因组.fasta文件
OUTPUT_EXCEL = "pangenome_mapping_results.xlsx"
IDENTITY_THRESHOLD = 90.0
COVERAGE_THRESHOLD = 80.0
NUM_THREADS = 32 # BLASTn内部线程数和并行处理菌株的线程池大小

def check_blast_installed():
    """检查BLAST+是否安装并可用"""
    try:
        subprocess.run(["blastn", "-version"], check=True, capture_output=True)
        subprocess.run(["makeblastdb", "-version"], check=True, capture_output=True)
        print("✅ BLAST+ 已安装并成功找到。")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 错误: 未找到 BLAST+。")
        print("请从 NCBI 官网下载并安装 BLAST+，并确保 'blastn' 和 'makeblastdb' 在系统的 PATH 中。")
        print("官网地址: https://www.ncbi.nlm.nih.gov/books/NBK279671/")
        return False

def create_blast_db(fasta_file):
    """为泛基因组参考文件创建BLAST数据库"""
    db_name = os.path.splitext(fasta_file)[0]
    # 检查数据库文件是否已存在，避免重复创建
    if os.path.exists(f"{db_name}.nsq"):
        print(f"ℹ️ 发现已存在的BLAST数据库 '{db_name}'，跳过创建步骤。")
        return db_name

    print(f"⚙️ 正在为 '{fasta_file}' 创建BLAST数据库...")
    command = [
        "makeblastdb",
        "-in", fasta_file,
        "-dbtype", "nucl",
        "-out", db_name
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ 成功创建BLAST数据库: '{db_name}'")
        return db_name
    except subprocess.CalledProcessError as e:
        print(f"❌ 创建BLAST数据库失败。")
        print(f"错误信息: {e.stderr}")
        sys.exit(1) # 退出程序

def get_query_lengths(ffn_file):
    """解析FFN文件，获取每个locus的长度"""
    try:
        return {record.id: len(record.seq) for record in SeqIO.parse(ffn_file, "fasta")}
    except FileNotFoundError:
        print(f"❌ 文件未找到: {ffn_file}")
        return {}

def process_strain(ffn_file, db_name):
    """对单个菌株FFN文件运行BLASTn并解析结果"""
    strain_name = os.path.basename(ffn_file).replace(".ffn", "")
    query_lengths = get_query_lengths(ffn_file)
    if not query_lengths:
        return strain_name, {}

    # 运行BLASTn
    blast_output = subprocess.run([
        "blastn",
        "-query", ffn_file,
        "-db", db_name,
        "-outfmt", "6 qseqid sseqid pident length bitscore",
        "-perc_identity", str(IDENTITY_THRESHOLD),
        "-num_threads", "1" # 每个并行任务内部使用单线程，由ThreadPoolExecutor控制并发
    ], capture_output=True, text=True, check=True).stdout

    # 解析结果
    best_hits = {}
    for line in blast_output.strip().split('\n'):
        if not line:
            continue
        parts = line.split('\t')
        qseqid, sseqid, pident, length, bitscore = parts[0], parts[1], float(parts[2]), int(parts[3]), float(parts[4])

        query_len = query_lengths.get(qseqid)
        if not query_len:
            continue
        
        coverage = (length / query_len) * 100.0

        if coverage >= COVERAGE_THRESHOLD:
            # 如果一个泛基因组基因有多个locus命中，选择bitscore最高的那个
            if sseqid not in best_hits or bitscore > best_hits[sseqid][1]:
                best_hits[sseqid] = (qseqid, bitscore)

    # 清理结果，只保留 locus 名称
    final_mapping = {sseqid: qseqid for sseqid, (qseqid, bitscore) in best_hits.items()}
    return strain_name, final_mapping

def main():
    """主执行函数"""
    print("--- 泛基因组映射流程开始 ---")
    
    # 1. 检查环境
    if not check_blast_installed():
        return

    if not os.path.exists(PANGENOME_FASTA):
        print(f"❌ 错误: 泛基因组文件 '{PANGENOME_FASTA}' 不存在。")
        return

    if not os.path.isdir(ANNOTATION_DIR):
        print(f"❌ 错误: 注释文件夹 '{ANNOTATION_DIR}' 不存在。")
        return

    # 2. 创建BLAST数据库
    db_name = create_blast_db(PANGENOME_FASTA)

    # 3. 获取所有菌株文件
    ffn_files = [os.path.join(ANNOTATION_DIR, f) for f in os.listdir(ANNOTATION_DIR) if f.endswith(".ffn")]
    if not ffn_files:
        print(f"❌ 错误: 在 '{ANNOTATION_DIR}' 文件夹中没有找到.ffn文件。")
        return
    
    print(f"🔍 找到 {len(ffn_files)} 个菌株 (.ffn) 文件待处理。")

    # 4. 使用多线程处理所有菌株
    all_results = {}
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # 提交所有任务
        future_to_strain = {executor.submit(process_strain, ffn, db_name): ffn for ffn in ffn_files}
        
        # 使用tqdm显示进度条
        progress_bar = tqdm(as_completed(future_to_strain), total=len(ffn_files), desc="🧬 BLASTn 比对中", unit=" 菌株")
        
        for future in progress_bar:
            try:
                strain_name, mapping = future.result()
                if mapping:
                    all_results[strain_name] = mapping
            except Exception as e:
                strain_ffn = future_to_strain[future]
                print(f"\n⚠️ 处理文件 {strain_ffn} 时发生错误: {e}")

    # 5. 整合结果并生成Excel
    print("\n⚙️ 所有比对完成，正在生成Excel报告...")
    if not all_results:
        print("⚠️ 没有找到任何符合阈值的匹配，无法生成Excel文件。")
        return

    # 创建DataFrame
    df = pd.DataFrame(all_results)
    
    # 获取完整的泛基因组基因列表作为索引
    pan_gene_names = [record.id for record in SeqIO.parse(PANGENOME_FASTA, "fasta")]
    df = df.reindex(index=pan_gene_names)
    
    # 将NaN值替换为空字符串
    df.fillna("", inplace=True)
    
    # 设置第一列的名称
    df.index.name = "Pan-gene"

    # 保存到Excel
    try:
        df.to_excel(OUTPUT_EXCEL)
        print(f"✅ 成功！结果已保存到 '{OUTPUT_EXCEL}'。")
    except Exception as e:
        print(f"❌ 保存Excel文件失败: {e}")

    print("--- 流程结束 ---")

if __name__ == "__main__":
    main()