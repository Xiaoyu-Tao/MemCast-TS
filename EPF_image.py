# filename: run_etth1_batch.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from main.EPF.EPF_main_vison_to_text import EPF_main_vison_to_text
from tqdm import tqdm

data_name = 'NP'
look_back, pred_window = 168, 24
api_key = 'sk-PxM40luD13UVKLhp6k3zenHC2XPASEi5uazXuXsCfTrQ3hUQ'  # 换成你的 key
max_workers = 5    # 并发线程数

def run_task(i, attr):
    # 可选：打印开始/结束（并发多时建议只用进度条）
    # print(f"Start: {attr}-{i}")
    EPF_main_vison_to_text(
        data_name, attr, look_back, pred_window, i, api_key=api_key,
        temperature=0.6,
        top_p=0.7
    )
    # print(f"Done:  {attr}-{i}")

if __name__ == "__main__":
    attrs = ['OT']
    tasks = [(i, attr) for i in range(30) for attr in attrs]  # 共 70 个任务

    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_task, *p) for p in tasks]
        # 使用 as_completed + tqdm 显示整体进度
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="EFP jobs", ncols=100):
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
