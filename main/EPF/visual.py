import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _local_extrema(y, order=3):
    """返回局部极大值与极小值的索引（简单邻域法）。"""
    n = len(y)
    peaks, troughs = [], []
    for i in range(order, n - order):
        left = y[i - order:i]
        right = y[i + 1:i + 1 + order]
        if y[i] > max(left) and y[i] > max(right):
            peaks.append(i)
        if y[i] < min(left) and y[i] < min(right):
            troughs.append(i)
    return np.array(peaks, dtype=int), np.array(troughs, dtype=int)


def plot_np_by_plan(df, timestamps, plan, look_back, pred_window,
                    current_cov_text=None, future_cov_text=None,
                    extrema_order=6, figsize=(11, 4.5),
                    y_pred=None, y_true_future=None):
    """绘制 Nord Pool 时序样本"""
    ts = pd.to_datetime(pd.Series(timestamps))
    df = df.copy()
    assert len(ts) == len(df)

    price_col, load_col, wind_col = 'Price', 'Load', 'Wind'
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # 历史真实价格
    ax1.plot(ts.iloc[:look_back], df[price_col].iloc[:look_back], label='Price (history)')

    # 预测窗口高亮
    if plan.get("show_future_window", True):
        ax1.axvspan(ts.iloc[look_back], ts.iloc[look_back + pred_window - 1],
                    alpha=0.12, label='Forecast Horizon')

    # 协变量
    ax2.plot(ts, df[load_col], linestyle='-', alpha=0.7, label='Grid Load')
    ax2.plot(ts, df[wind_col], linestyle='--', alpha=0.7, label='Wind Power')

    # 未来预测与真实值
    if y_pred is not None:
        ax1.plot(ts.iloc[look_back: look_back + pred_window], y_pred,
                 linestyle='--', label='Predicted Price')
    if y_true_future is not None:
        ax1.plot(ts.iloc[look_back: look_back + pred_window], y_true_future,
                 alpha=0.6, label='True Price (future, eval only)')

    # 周末标注
    if "weekday/weekend shading" in plan.get("add_annotations", []):
        dow = ts.dt.dayofweek.values
        in_weekend = (dow >= 5)
        start = None
        for i in range(len(ts)):
            if in_weekend[i] and start is None:
                start = ts.iloc[i]
            if (not in_weekend[i] and start is not None) or (i == len(ts)-1 and start is not None):
                end = ts.iloc[i]
                ax1.axvspan(start, end, alpha=0.08)
                start = None

    # 局部极值
    if "local extrema markers" in plan.get("add_annotations", []):
        y_hist = df[price_col].iloc[:look_back].values
        peaks, troughs = _local_extrema(y_hist, order=extrema_order)
        ax1.scatter(ts.iloc[peaks], y_hist[peaks], marker='^', s=30, label='Local Max')
        ax1.scatter(ts.iloc[troughs], y_hist[troughs], marker='v', s=30, label='Local Min')

    # 协变量文本
    if current_cov_text:
        ax1.text(ts.iloc[int(0.1 * len(ts))], np.nanmax(df[price_col].iloc[:look_back]) * 0.95,
                 f"Current: {current_cov_text}", fontsize=8)
    if future_cov_text:
        ax1.text(ts.iloc[-int(0.3 * len(ts))], np.nanmax(df[price_col].iloc[:look_back]) * 0.88,
                 f"Future: {future_cov_text}", fontsize=8)

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price (€/MWh)')
    ax2.set_ylabel('Load / Wind')

    title = "NP: Multi-axis temporal overlay (Price vs Load/Wind)"
    if plan.get("rationale"):
        title += "\n" + plan["rationale"]
    ax1.set_title(title)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def generate_covariates_text(data, look_back, pred_window):
    """根据负荷/风电/电价动态生成描述文本"""
    current_data = data.iloc[look_back - 1]
    load_level = "high" if current_data['Load'] > 30000 else "low" if current_data['Load'] < 15000 else "medium"
    wind_level = "high" if current_data['Wind'] > 1000 else "low" if current_data['Wind'] < 300 else "medium"
    price_level = "high" if current_data['Price'] > 60 else "low" if current_data['Price'] < 30 else "medium"

    current_cov_text = f"current day type: {'weekday' if current_data['date'].weekday() < 5 else 'weekend'}, load={load_level}, wind={wind_level}, price={price_level}"

    future_data = data.iloc[look_back: look_back + pred_window]
    future_load = future_data['Load'].mean()
    future_wind = future_data['Wind'].mean()
    future_load_level = "high" if future_load > 30000 else "low" if future_load < 15000 else "medium"
    future_wind_level = "high" if future_wind > 1000 else "low" if future_wind < 300 else "medium"

    future_cov_text = f"future day type: {'weekend' if future_data['date'].iloc[0].weekday() >= 5 else 'weekday'}, load={future_load_level}, wind={future_wind_level}"

    return current_cov_text, future_cov_text


if __name__ == "__main__":
    # === 参数 ===
    data_name = 'NP'
    data = pd.read_csv(f'/data/songliv/TS/datasets/EPF/{data_name}.csv')
    look_back, pred_window = 168, 24
    stride = 24
    num_samples = 30
    out_dir = f"/data/songliv/TS/TimeReasoner/output/EPF/{data_name}"
    os.makedirs(out_dir, exist_ok=True)

    # === 数据处理 ===
    start_idx = int(len(data) * 0.8) - look_back
    if start_idx < 0:
        start_idx = 0
    data = data.iloc[start_idx:].reset_index(drop=True)
    data['date'] = pd.to_datetime(data['date'])
    rename_map = {
        'Grid load forecast': 'Load',
        'Wind power forecast': 'Wind',
        'OT': 'Price'
    }
    data.columns = data.columns.str.strip()
    data = data.rename(columns=rename_map)

    # === 可视化方案 ===
    plan = {
        "visualization_type": "multi-axis temporal overlay",
        "main_variable": "Electricity Price",
        "secondary_variables": ["Grid Load", "Wind Power"],
        "show_future_window": True,
        "add_annotations": ["weekday/weekend shading", "forecast horizon highlight", "local extrema markers"],
        "rationale": "A multi-axis temporal overlay allows the VLM to align price variations with load/wind over time."
    }

    end = look_back + pred_window
    max_start = len(data) - end + 1

    # === 滑动窗口绘制 ===
    for idx, start in enumerate(range(0, max_start, stride)):
        if idx >= num_samples:
            break
        win = data.iloc[start:start + end].copy()
        current_cov_text, future_cov_text = generate_covariates_text(win, look_back, pred_window)
        fig = plot_np_by_plan(
            df=win[['Price', 'Load', 'Wind']],
            timestamps=win['date'],
            plan=plan,
            look_back=look_back,
            pred_window=pred_window,
            current_cov_text=current_cov_text,
            future_cov_text=future_cov_text
        )

        # 保存图像
        out_path = os.path.join(out_dir, f"{data_name}_visualization_{idx:03d}.png")
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 保存元信息 JSON
        meta = {
            "index": idx,
            "start_index": start,
            "start_date": str(win['date'].iloc[0]),
            "end_date": str(win['date'].iloc[-1]),
            "current_cov_text": current_cov_text,
            "future_cov_text": future_cov_text,
            "look_back": look_back,
            "pred_window": pred_window,
            "stride": stride
        }
        with open(out_path.replace('.png', '.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    print(f"✅ Generated {num_samples} samples (stride={stride}) → {out_dir}")
