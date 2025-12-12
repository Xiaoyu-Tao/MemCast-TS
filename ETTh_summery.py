# filename: run_etth1_batch.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from main.ETTh.ETTh_main_summery import analyze_memory_reasoning
from tqdm import tqdm

data_name = 'ETTh1'
look_back, pred_window = 96, 96
api_key = 'sk-PxM40luD13UVKLhp6k3zenHC2XPASEi5uazXuXsCfTrQ3hUQ'  # 换成你的 key
max_workers = 20     # 并发线程数

def run_task(i, attr):
    # 可选：打印开始/结束（并发多时建议只用进度条）
    # print(f"Start: {attr}-{i}")
    analyze_memory_reasoning(
        data_name=data_name,
        api_key=api_key,
        attr=attr,
        look_back=look_back,
        pred_window=pred_window,
        output_dir='/data/songliv/TS/TimeReasoner/Memory/summary_outputs',
        overwrite=False,
        number=i
    )

if __name__ == "__main__":
    attrs = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    tasks = [(i, attr) for i in range(118) for attr in attrs]  # 共 70 个任务

    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_task, *p) for p in tasks]
        # 使用 as_completed + tqdm 显示整体进度
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="ETTh1 jobs", ncols=100):
            try:
                fut.result()  # 取结果并抛出异常（若有）
            except Exception as e:
                errors.append(str(e))

    # 执行完的简要汇总
    done_cnt = len(futures) - len(errors)
    print(f"\n✅ finished: {done_cnt}/{len(futures)}")
    if errors:
        print(f"❌ errors: {len(errors)} (showing first 5)")
        for msg in errors[:5]:
            print("  -", msg)
