import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
GLOBAL_TRAINVAL_DF = None

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
                    y_pred=None, y_true_future=None, col_names=None):
    ts = pd.to_datetime(pd.Series(timestamps))
    df = df.copy()
    assert len(ts) == len(df)

    if col_names is None:
        price_col, load_col, wind_col = 'OT', 'HUFL', 'HULL'
    else:
        price_col = col_names.get('target', 'OT')
        load_col = col_names.get('cov1', 'HUFL')
        wind_col = col_names.get('cov2', 'HULL')
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    ax1.plot(ts.iloc[:look_back], df[price_col].iloc[:look_back], label=f'{price_col} (history)')

    # 预测窗口高亮
    if plan.get("show_future_window", True):
        ax1.axvspan(ts.iloc[look_back], ts.iloc[look_back + pred_window - 1],
                    alpha=0.12, label='Forecast Horizon')

    ax2.plot(ts, df[load_col], linestyle='-', alpha=0.7, label=load_col)
    ax2.plot(ts, df[wind_col], linestyle='--', alpha=0.7, label=wind_col)

    if y_pred is not None:
        ax1.plot(ts.iloc[look_back: look_back + pred_window], y_pred,
                 linestyle='--', label=f'Predicted {price_col}')
    if y_true_future is not None:
        ax1.plot(ts.iloc[look_back: look_back + pred_window], y_true_future,
                 alpha=0.6, label=f'True {price_col} (future, eval only)')

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
    ax1.set_ylabel(price_col)
    ax2.set_ylabel(f'{load_col} / {wind_col}')

    title = f"ETTh1: {price_col} vs {load_col}/{wind_col}"
    if plan.get("rationale"):
        title += "\n" + plan["rationale"]
    ax1.set_title(title)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def generate_covariates_text(data, look_back, pred_window, col_names):
    current_data = data.iloc[look_back - 1]
    cov1 = col_names.get('cov1', 'HUFL')
    cov2 = col_names.get('cov2', 'HULL')
    target = col_names.get('target', 'OT')
    q1_cov1 = data[cov1].quantile(0.33)
    q2_cov1 = data[cov1].quantile(0.66)
    q1_cov2 = data[cov2].quantile(0.33)
    q2_cov2 = data[cov2].quantile(0.66)
    q1_target = data[target].quantile(0.33)
    q2_target = data[target].quantile(0.66)
    def _lvl(v, a, b):
        return "low" if v < a else "high" if v > b else "medium"
    cov1_level = _lvl(current_data[cov1], q1_cov1, q2_cov1)
    cov2_level = _lvl(current_data[cov2], q1_cov2, q2_cov2)
    target_level = _lvl(current_data[target], q1_target, q2_target)
    current_cov_text = f"current day type: {'weekday' if current_data['date'].weekday() < 5 else 'weekend'}, {cov1}={cov1_level}, {cov2}={cov2_level}, {target}={target_level}"
    future_data = data.iloc[look_back: look_back + pred_window]
    future_cov1 = future_data[cov1].mean()
    future_cov2 = future_data[cov2].mean()
    f1 = "high" if future_cov1 > q2_cov1 else "low" if future_cov1 < q1_cov1 else "medium"
    f2 = "high" if future_cov2 > q2_cov2 else "low" if future_cov2 < q1_cov2 else "medium"
    future_cov_text = f"future day type: {'weekend' if future_data['date'].iloc[0].weekday() >= 5 else 'weekday'}, {cov1}={f1}, {cov2}={f2}"
    return current_cov_text, future_cov_text

def pick_covariates_by_pearson(win_df, target_col='OT', candidate_cols=None):
    if 'tmax' in win_df.columns and 'tmin' in win_df.columns:
        return ('tmax', 'tmin')
    return ('HUFL', 'HULL')
    # if candidate_cols is None:
    #     candidate_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    # df = GLOBAL_TRAINVAL_DF if GLOBAL_TRAINVAL_DF is not None else win_df
    # present = [c for c in candidate_cols if c in df.columns and c != target_col]
    # if target_col not in df.columns or len(present) == 0:
    #     return ('HUFL', 'HULL')
    # target = pd.to_numeric(df[target_col], errors='coerce')
    # best = {}
    # for cov in present:
    #     cov_series = pd.to_numeric(df[cov], errors='coerce')
    #     pair = pd.concat([target, cov_series], axis=1).dropna()
    #     if len(pair) >= 2:
    #         val = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    #         if val is not None and not np.isnan(val):
    #             best[cov] = abs(val)
    #         else:
    #             best[cov] = np.nan
    #     else:
    #         best[cov] = np.nan
    # scores = pd.Series(best).dropna()
    # if scores.empty:
    #     return ('HUFL', 'HULL') if len(present) >= 2 else (present[0], present[0])
    # order = scores.sort_values(ascending=False).index.tolist()
    # cov1 = order[0]
    # cov2 = order[1] if len(order) > 1 else (present[1] if len(present) > 1 and present[1] != cov1 else present[0])
    # return (cov1, cov2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate ETTh time-series visualizations')
    parser.add_argument('--mode', choices=['memory', 'test'], default='memory',
                        help="'memory': last N training samples; 'test': first N test samples")
    args = parser.parse_args()

    # === 参数 ===
    data_name = 'ETTh1'
    df = pd.read_csv(f'./datasets/{data_name}.csv')
    look_back, pred_window = 96, 96
    stride = 24
    num_samples = 30
    out_dir = f"./Memory/visual/image/ETTh/{data_name}"
    os.makedirs(out_dir, exist_ok=True)

    # === 数据处理 ===
    df.columns = df.columns.str.strip()
    df['date'] = pd.to_datetime(df['date'])
    split_idx = int(len(df) * 0.8)
    end = look_back + pred_window

    if args.mode == 'memory':
        # 训练集：取最后 num_samples 个窗口，global_idx = start（data_slice 从 df 第 0 行起）
        data_offset = 0
        data_slice = df.iloc[:split_idx].reset_index(drop=True)
        max_start = len(data_slice) - end + 1
        starts = list(range(0, max_start, stride))
        selected = starts[-num_samples:]
    else:  # test
        # 测试集：取最前 num_samples 个窗口，global_idx = test_start + start
        test_start = max(split_idx - look_back, 0)
        data_offset = test_start
        data_slice = df.iloc[test_start:].reset_index(drop=True)
        max_start = len(data_slice) - end + 1
        starts = list(range(0, max_start, stride))
        selected = starts[:num_samples]

    GLOBAL_TRAINVAL_DF = data_slice

    # === 可视化方案 ===
    plan = {
        "visualization_type": "multi-axis temporal overlay",
        "main_variable": "OT",
        "secondary_variables": ["tmax", "tmin"],
        "show_future_window": True,
        "add_annotations": ["weekday/weekend shading", "forecast horizon highlight", "local extrema markers"],
        "rationale": "Overlay helps align target variations with covariates over time."
    }

    # === 滑动窗口绘制 ===
    global_cov1, global_cov2 = pick_covariates_by_pearson(data_slice, target_col='OT')
    for idx, start in enumerate(selected):
        win = data_slice.iloc[start:start + end].copy()
        c1, c2 = global_cov1, global_cov2
        col_names = {"target": "OT", "cov1": c1, "cov2": c2}
        current_cov_text, future_cov_text = generate_covariates_text(win, look_back, pred_window, col_names)
        fig = plot_np_by_plan(
            df=win[[col_names["target"], col_names["cov1"], col_names["cov2"]]],
            timestamps=win['date'],
            plan=plan,
            look_back=look_back,
            pred_window=pred_window,
            current_cov_text=current_cov_text,
            future_cov_text=future_cov_text,
            col_names=col_names
        )

        # 保存图像（使用样本顺序索引 idx，memory/test 分子目录）
        mode_dir = os.path.join(out_dir, args.mode)
        os.makedirs(mode_dir, exist_ok=True)
        out_path = os.path.join(mode_dir, f"{data_name}_visualization_{idx:03d}.png")
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 保存元信息 JSON
        global_idx = data_offset + start
        meta = {
            "index": idx,
            "global_idx": global_idx,
            "start_index": start,
            "start_date": str(win['date'].iloc[0]),
            "end_date": str(win['date'].iloc[-1]),
            "current_cov_text": current_cov_text,
            "future_cov_text": future_cov_text,
            "look_back": look_back,
            "pred_window": pred_window,
            "stride": stride,
            "target": col_names["target"],
            "covariates": [col_names["cov1"], col_names["cov2"]]
        }
        with open(out_path.replace('.png', '.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    print(f"✅ Generated {len(selected)} {args.mode} samples (stride={stride}) → {out_dir}/{args.mode}")
