# filename: run_etth1_batch.py
from main.ETTh.ETTh_main_one_shot_reasoning import ETTh_main_one_shot_reasoning

data_name = 'ETTh1'
look_back, pred_window = 96, 96
api_key = 'sk-PxM40luD13UVKLhp6k3zenHC2XPASEi5uazXuXsCfTrQ3hUQ'  # 你的 key

if __name__ == "__main__":
    # 只运行一个样本，例如 attr='HUFL', number=0
    attr = 'HUFL'
    number = 0

    print(f"🔍 Testing single sample: attr={attr}, number={number}")
    ETTh_main_one_shot_reasoning(
        data_name=data_name,
        attr=attr,
        look_back=look_back,
        pred_window=pred_window,
        number=number,
        api_key=api_key,
        temperature=0.6,
        top_p=0.7
    )
    print("✅ Single sample test completed.")
