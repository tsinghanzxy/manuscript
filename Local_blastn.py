import os
import subprocess
import glob
from Bio import SeqIO
from Bio.Blast import NCBIXML
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import time
from threading import Lock
import atexit
import psutil
import matplotlib.pyplot as plt

# 配置参数
QUERY_FILE = "lag1.fasta"#此处可替换为lpg0773用以检测菌株是否为sg1
STRAIN_FOLDER = "4147_LP_fna"
THREADS = os.cpu_count()
MIN_COVERAGE = 50  #初始使用低阈值，实际结果查看.xlsx文件，可进一步调整阈值
MIN_IDENTITY = 50
CHUNK_SIZE = 100  # 每处理100个文件保存一次中间结果
MEMORY_THRESHOLD = 85  # 内存使用百分比告警阈值
LOG_FILE = "processed.log"  # 断点续传记录文件

# 全局状态
total_files = 0
processed_files = 0
progress_lock = Lock()
start_time = time.time()
results = []
results_lock = Lock()
db_files = set()
cleanup_lock = Lock()
processed_set = set()  # 已处理文件集合

# 注册退出清理函数
def cleanup_resources():
    with cleanup_lock:
        print("\n正在清理临时文件...")
        patterns = ["*.nhr", "*.nin", "*.nsq", "*_db", "*_blast.xml"]
        for pattern in patterns:
            for f in glob.glob(os.path.join(STRAIN_FOLDER, pattern)):
                try:
                    os.remove(f)
                except:
                    pass
        # 验证清理结果
        remaining = list(glob.glob(os.path.join(STRAIN_FOLDER, "*.nhr")))
        if remaining:
            print(f"警告：残留数据库文件 {len(remaining)} 个")

atexit.register(cleanup_resources)

def save_checkpoint(filename):
    """保存处理进度"""
    with open(LOG_FILE, "a") as f:
        f.write(f"{filename}\n")

def load_checkpoint():
    """加载断点续传记录"""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return set(line.strip() for line in f)
    return set()

def save_intermediate_results():
    """保存中间结果"""
    if not results:
        return
    timestamp = int(time.time())
    temp_file = f"temp_results_{timestamp}.pkl"
    with results_lock:
        df = pd.DataFrame(results)
        df.to_pickle(temp_file)
        results.clear()
    print(f"\n已保存中间结果到 {temp_file}")

def process_file(filename):
    global processed_files
    strain_name = os.path.splitext(filename)[0]
    filepath = os.path.join(STRAIN_FOLDER, filename)
    hits = []
    db_name = ""

    try:
        # 内存检查
        if psutil.virtual_memory().percent > MEMORY_THRESHOLD:
            print(f"\n内存使用超过{MEMORY_THRESHOLD}%!")
            save_intermediate_results()

        # 创建BLAST数据库
        db_name = os.path.splitext(filepath)[0] + "_db"
        with cleanup_lock:
            db_files.add(db_name)
        
        db_cmd = f"makeblastdb -in {filepath} -dbtype nucl -out {db_name} > /dev/null 2>&1"
        subprocess.run(db_cmd, shell=True, check=True)

        # 运行BLAST
        blast_output = f"{strain_name}_blast.xml"
        blast_cmd = (
            f"blastn -query {QUERY_FILE} -db {db_name} "
            f"-outfmt 5 -out {blast_output} > /dev/null 2>&1"
        )
        subprocess.run(blast_cmd, shell=True, check=True)

        # 解析结果
        with open(blast_output, "r") as blast_file:
            blast_records = NCBIXML.parse(blast_file)
            
            for blast_record in blast_records:
                for alignment in blast_record.alignments:
                    for hsp in alignment.hsps:
                        coverage = (hsp.query_end - hsp.query_start + 1) / query_length * 100
                        identity = (hsp.identities / hsp.align_length) * 100
                        
                        if coverage >= MIN_COVERAGE and identity >= MIN_IDENTITY:
                            hits.append({
                                "Strain": strain_name,
                                "Target Sequence": alignment.hit_id,
                                "Identity (%)": round(identity, 2),
                                "Coverage (%)": round(coverage, 2),
                                "E-value": f"{hsp.expect:.2e}",
                                "Query Start": hsp.query_start,
                                "Query End": hsp.query_end,
                                "Subject Start": hsp.sbjct_start,
                                "Subject End": hsp.sbjct_end
                            })

        # 立即清理临时文件
        os.remove(blast_output)
        for ext in [".nhr", ".nin", ".nsq"]:
            db_file = db_name + ext
            if os.path.exists(db_file):
                os.remove(db_file)
                with cleanup_lock:
                    db_files.discard(db_name)

    except subprocess.CalledProcessError as e:
        print(f"\n处理 {filename} 时发生错误：{str(e)}")
    except Exception as e:
        print(f"\n解析 {filename} 时出现异常：{str(e)}")
    finally:
        # 确保删除残留文件
        if db_name:
            for ext in [".nhr", ".nin", ".nsq"]:
                db_file = db_name + ext
                if os.path.exists(db_file):
                    os.remove(db_file)
                    with cleanup_lock:
                        db_files.discard(db_name)
        
        with progress_lock:
            processed_files += 1
            if processed_files % CHUNK_SIZE == 0:
                save_intermediate_results()
        
        save_checkpoint(filename)
        return hits

def update_progress(pbar):
    while True:
        with progress_lock:
            current = processed_files
        if current >= total_files:
            break
        elapsed = time.time() - start_time
        speed = current / elapsed if elapsed > 0 else 0
        remaining = (total_files - current) / speed if speed > 0 else 0
        pbar.set_postfix({
            '速度': f'{speed:.1f} 文件/秒',
            '剩余时间': f'{remaining:.1f}秒',
            '已处理': f'{current}/{total_files}',
            '内存': f'{psutil.virtual_memory().percent}%'
        })
        pbar.update(current - pbar.n)
        time.sleep(0.2)

def merge_temp_files():
    """合并所有临时结果文件"""
    temp_files = glob.glob("temp_results_*.pkl")
    final_df = pd.DataFrame()
    
    for f in temp_files:
        try:
            df = pd.read_pickle(f)
            final_df = pd.concat([final_df, df])
            os.remove(f)
        except Exception as e:
            print(f"合并文件{f}时出错：{str(e)}")
    
    return final_df

def generate_visualization(df):
    """生成可视化图表"""
    plt.figure(figsize=(12, 6))
    
    # Identity分布
    plt.subplot(1, 2, 1)
    df['Identity (%)'].hist(bins=20, color='skyblue')
    plt.title('序列相似度分布')
    plt.xlabel('Identity (%)')
    plt.ylabel('频数')
    
    # Coverage分布
    plt.subplot(1, 2, 2)
    df['Coverage (%)'].hist(bins=20, color='lightgreen')
    plt.title('序列覆盖度分布')
    plt.xlabel('Coverage (%)')
    plt.ylabel('频数')
    
    plt.tight_layout()
    plt.savefig('blast_analysis.png')
    print("已生成可视化图表：blast_analysis.png")

def main():
    global total_files, processed_set
    
    # 读取查询序列
    try:
        query_seq = next(SeqIO.parse(QUERY_FILE, "fasta"))
        query_length = len(query_seq.seq)
    except Exception as e:
        print(f"无法读取查询文件：{str(e)}")
        return

    # 加载断点记录
    processed_set = load_checkpoint()
    print(f"找到 {len(processed_set)} 个已处理记录")

    # 获取任务列表
    try:
        all_files = [f for f in os.listdir(STRAIN_FOLDER)
                    if f.lower().endswith((".fasta", ".fna"))]
        files = [f for f in all_files if f not in processed_set]
    except FileNotFoundError:
        print(f"目录 {STRAIN_FOLDER} 不存在")
        return
    
    total_files = len(files)
    if total_files == 0:
        print("没有需要处理的新文件")
        return

    # 进度条配置
    progress_desc = f"BLAST分析（{THREADS}线程）"
    with tqdm(total=total_files, desc=progress_desc, unit="文件", 
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [剩余: {remaining}]") as pbar:
        
        # 启动进度更新线程
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as progress_executor:
            progress_future = progress_executor.submit(update_progress, pbar)
            
            # 处理线程池
            with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
                futures = [executor.submit(process_file, f) for f in files]
                
                for future in concurrent.futures.as_completed(futures):
                    hits = future.result()
                    if hits:
                        with results_lock:
                            results.extend(hits)
            
            # 等待进度更新完成
            progress_future.result()

    # 合并结果
    final_df = merge_temp_files()
    if not results and final_df.empty:
        print("\n没有找到符合条件的结果")
        return
    
    if not final_df.empty:
        with results_lock:
            final_df = pd.concat([final_df, pd.DataFrame(results)])
    
    # 生成最终报告
    column_order = [
        "Strain", "Target Sequence", "Identity (%)", "Coverage (%)",
        "E-value", "Query Start", "Query End", "Subject Start", "Subject End"
    ]
    final_df = final_df[column_order].sort_values(
        by=["Strain", "Identity (%)", "Coverage (%)"], 
        ascending=[True, False, False]
    )
    
    # 保存Excel
    writer = pd.ExcelWriter(
        "lag1_blast_results.xlsx",
        engine='xlsxwriter',
        engine_kwargs={'options': {'strings_to_numbers': True}}
    )
    final_df.to_excel(writer, index=False)
    
    # 调整列宽
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    for idx, col in enumerate(final_df.columns):
        max_len = max(final_df[col].astype(str).map(len).max(), len(col)) + 2
        worksheet.set_column(idx, idx, max_len)
    
    writer.close()
    
    # 生成可视化
    generate_visualization(final_df)
    print(f"\n成功处理 {len(final_df)} 条记录，结果已保存")

if __name__ == "__main__":
    main()