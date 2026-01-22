from concurrent.futures import ThreadPoolExecutor, as_completed
from main.EPF.EPF_main_one_shot_reasoning import EPF_main_one_shot_reasoning  
from main.EPF.EPF_main_one_shot_reasoning_OT import EPF_main_one_shot_reasoning_OT
from tqdm import tqdm

data_name = 'NP'
look_back, pred_window = 168, 24
max_workers = 10

def run_task(i, attr):
    EPF_main_one_shot_reasoning_OT(
        data_name, attr, look_back, pred_window, i,
        temperature=0.6,
        top_p=0.7
    )

if __name__ == "__main__":
    attrs = ['OT']
    tasks = [(i, attr) for i in range(30) for attr in attrs]

    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_task, *p) for p in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="EFP jobs", ncols=100):
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
