from concurrent.futures import ThreadPoolExecutor, as_completed
from main.ETTh.ETTh_main_summery import analyze_memory_reasoning
from tqdm import tqdm

data_name = 'ETTh1'
look_back, pred_window = 96, 96
max_workers = 20

def run_task(i, attr):
    analyze_memory_reasoning(
        data_name=data_name,
        attr=attr,
        look_back=look_back,
        pred_window=pred_window,
        output_dir='Memory/summary_outputs',
        overwrite=False,
        number=i
    )

if __name__ == "__main__":
    attrs = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    tasks = [(i, attr) for i in range(118) for attr in attrs]

    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_task, *p) for p in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="ETTh1 jobs", ncols=100):
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
