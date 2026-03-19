# filename: run_etth1_batch.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from main.ETTh.ETTh_main_vison_to_text import ETTh_main_vison_to_text
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()

data_name = 'ETTh1'
look_back, pred_window = 96, 96
api_key = os.getenv("OPENAI_API_KEY")
max_workers = 20    # 并发线程数

def run_task(global_idx, attr, mode):
    ETTh_main_vison_to_text(
        data_name, attr, look_back, pred_window, global_idx, api_key=api_key,
        temperature=0.6,
        top_p=0.7,
        mode=mode
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ETTh vision-to-text batch runner')
    parser.add_argument('--mode', choices=['memory', 'test'], default='memory',
                        help="'memory': last N training samples; 'test': first N test samples")
    args = parser.parse_args()

    attrs = ['OT']
    num_samples = 30
    global_indices = list(range(num_samples))

    tasks = [(g, attr, args.mode) for attr in attrs for g in global_indices]

    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_task, *p) for p in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"ETTh {args.mode} jobs", ncols=100):
            try:
                fut.result()
            except Exception as e:
                errors.append(str(e))

    done_cnt = len(futures) - len(errors)
    print(f"\n✅ finished: {done_cnt}/{len(futures)}")
    if errors:
        print(f"❌ errors: {len(errors)} (showing first 5)")
        for msg in errors[:5]:
            print("  -", msg)
