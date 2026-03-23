# filename: run_etth1_batch.py
from concurrent.futures import ThreadPoolExecutor, as_completed
# from main.ETTh.ETTh_main_one_shot_reasoning_see import ETTh_main_one_shot_reasoning_see
from main.ETTh.ETTh_main_few_shot_reasoning import ETTh_main_few_shot_reasoning
from tqdm import tqdm
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

data_name = 'ETTh1'
look_back, pred_window = 96, 96
api_key = os.getenv("OPENAI_API_KEY")
max_workers = 5    # 并发线程数

def run_task(i, attr):
    # 可选：打印开始/结束（并发多时建议只用进度条）
    # print(f"Start: {attr}-{i}")
    ETTh_main_few_shot_reasoning(
        data_name, attr, look_back, pred_window, i, api_key=api_key,
        temperature=0.6,
        top_p=0.7,
        use_summary=True
    )
    # print(f"Done:  {attr}-{i}")

if __name__ == "__main__":
    attrs = ['OT']
    data_path = f'./datasets/{data_name}.csv'
    df = pd.read_csv(data_path)
    start_idx = int(len(df) * 0.8) - look_back
    end_idx = int(len(df))
    end = look_back + pred_window
    stride = 24
    max_start = end_idx - end + 1
    starts = list(range(start_idx, max_start, stride))
    sample_count = len(starts)
    # sample_count = (max_start + stride - 1) // stride if max_start > 0 else 0
    # last_n = 30
    # start_boundary = max(sample_count - last_n, 0)
    # tasks = [(i, attr) for attr in attrs for i in range(start_boundary, sample_count)]
    num_samples=30
    tasks = [(i, attr) for attr in attrs for i in range(0, num_samples)]
    # tasks = [(18, 'OT')]
    # tasks = [(2, 'OT')]

    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_task, *p) for p in tasks]
        # 使用 as_completed + tqdm 显示整体进度
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="ETTh jobs", ncols=100):
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
