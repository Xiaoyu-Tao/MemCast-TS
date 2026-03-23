import os
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from utils.metrics import MSE, MAE
from utils.tools import visual
from utils.api_ouput import gpt4_api_output
from main.EPF.EPF_main_few_shot_reasoning import retrieve_similar_examples

def parse_prediction_from_answer(answer_text):
    """
    从 LLM 的 answer 文本中解析预测结果，兼容空格分隔或逗号分隔格式
    """
    dates, values = [], []
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, answer_text, re.DOTALL | re.IGNORECASE)

    if not match:
        return dates, values

    content = match.group(1).strip()

    if content.startswith("```"):
        content = re.sub(r'^```[^\n]*\n', '', content, count=1)
        content = re.sub(r'\n?```$', '', content, count=1)
        content = content.strip()

    lines = content.split('\n')

    for line in lines:
        line = line.strip()
        if not line or line.startswith(('date', '<', '>')):  # 跳过表头与标签
            continue

        # 尝试解析逗号分隔
        if ',' in line:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                date_str, val_str = parts[0], parts[1]
            else:
                continue
        else:
            # 尝试空格分隔格式
            parts = line.split()
            if len(parts) >= 3:
                date_str = parts[0] + ' ' + parts[1]
                val_str = parts[2]
            elif len(parts) == 2:
                date_str, val_str = parts
            else:
                continue

        try:
            value = float(val_str)
            dates.append(date_str)
            values.append(value)
        except ValueError:
            continue

    return dates, values


_summary_index_cache = {}
_summary_cache = {}
_historical_examples_cache = {}


def _summary_dir_for(data_name):
    return os.path.join("Memory", "cases", "summary", "EPF", data_name)


def _summary_path(data_name, attr, look_back, pred_window, index):
    return os.path.join(
        _summary_dir_for(data_name),
        f"result_{attr}_{data_name}_{look_back}_{pred_window}_{index}.json"
    )


def _get_summary_indices(data_name, attr, look_back, pred_window):
    key = (data_name, attr, look_back, pred_window)
    if key in _summary_index_cache:
        return _summary_index_cache[key]
    summary_dir = _summary_dir_for(data_name)
    prefix = f"result_{attr}_{data_name}_{look_back}_{pred_window}_"
    indices = set()
    if os.path.exists(summary_dir):
        for fn in os.listdir(summary_dir):
            if not (fn.startswith(prefix) and fn.endswith(".json")):
                continue
            idx_str = fn[len(prefix):-5]
            if idx_str.isdigit():
                indices.add(int(idx_str))
    _summary_index_cache[key] = indices
    return indices


def _load_summary(data_name, attr, look_back, pred_window, index):
    key = (data_name, attr, look_back, pred_window, int(index))
    if key in _summary_cache:
        return _summary_cache[key]
    path = _summary_path(data_name, attr, look_back, pred_window, index)
    if not os.path.exists(path):
        _summary_cache[key] = None
        return None
    with open(path, "r") as f:
        summary = json.load(f)
    _summary_cache[key] = summary
    return summary


def _build_distilled_block(item, case_type):
    parts = []
    if case_type == "good":
        experience = item.get("experience")
        success_insight = item.get("success_insight")
        if experience:
            parts.append(f"experience: {experience}")
        if success_insight:
            parts.append(f"success_insight: {success_insight}")
    else:
        failure_analysis = item.get("failure_analysis")
        preventative_rule = item.get("preventative_rule")
        if failure_analysis:
            parts.append(f"failure_analysis: {failure_analysis}")
        if preventative_rule:
            parts.append(f"preventative_rule: {preventative_rule}")
    meta_logic = item.get("meta_logic")
    if meta_logic is not None:
        meta_logic_str = json.dumps(meta_logic, ensure_ascii=False) if isinstance(meta_logic, dict) else str(meta_logic)
        parts.append(f"meta_logic: {meta_logic_str}")
    return "\n".join(parts).strip()


def _extract_distilled_blocks(summary):
    if not isinstance(summary, dict):
        return []
    blocks = []
    refined = summary.get("refined_distilled")
    if isinstance(refined, list) and refined:
        for item in refined[:3]:
            if not isinstance(item, dict):
                continue
            block = _build_distilled_block(item, "good")
            if block:
                blocks.append(block)
        if blocks:
            return blocks
    good_items = summary.get("good_cases_distilled") or []
    bad_items = summary.get("bad_cases_distilled") or []
    for case_type, case_items in [("good", good_items), ("bad", bad_items)]:
        if not case_items:
            continue
        case_entry = case_items[0] if isinstance(case_items, list) and case_items else None
        if not isinstance(case_entry, dict):
            continue
        distilled = case_entry.get("distilled")
        if not isinstance(distilled, list):
            continue
        for item in distilled[:3]:
            if not isinstance(item, dict):
                continue
            block = _build_distilled_block(item, case_type)
            if block:
                blocks.append(block)
    return blocks


def _build_historical_examples(data_name, attr, look_back, pred_window):
    key = (data_name, attr, look_back, pred_window)
    if key in _historical_examples_cache:
        return _historical_examples_cache[key]
    data_dir = f'./datasets/{data_name}.csv'
    data = pd.read_csv(data_dir)
    target_col = attr
    if target_col not in data.columns:
        candidates = [
            col for col in data.columns
            if attr.lower().strip() == col.lower().strip()
            or attr.lower().strip() in col.lower().strip()
            or col.lower().strip() in attr.lower().strip()
        ]
        if candidates:
            target_col = candidates[0]
    summary_indices = _get_summary_indices(data_name, attr, look_back, pred_window)
    slide_window = 24
    max_samples = (len(data) - look_back) // slide_window + 1
    historical_examples = []
    for i in range(max_samples):
        if summary_indices and i not in summary_indices:
            continue
        start_idx = i * slide_window
        end_idx = start_idx + look_back
        if end_idx > len(data):
            continue
        segment = data.iloc[start_idx:end_idx]
        values = segment[target_col].values
        if len(values) == 0 or pd.isna(values).all():
            continue
        historical_examples.append({
            "data": values,
            "index": i
        })
    _historical_examples_cache[key] = historical_examples
    return historical_examples


def _format_trajectory_values(values):
    formatted = []
    for v in values:
        if v is None or not np.isfinite(v):
            formatted.append("nan")
        else:
            formatted.append(f"{float(v):.6f}")
    return ", ".join(formatted)


def _build_llm_selection_prompt(distilled_blocks, trajectories):
    """
    Builds a prompt for an LLM to select the best time-series trajectory
    based on historical distilled experiences.
    """
    prompt = (
        "Role: You are a world-class Expert in Time Series Forecasting and Evaluation.\n"
        "Task: Your objective is to analyze multiple candidate trajectories (predictions) "
        "and determine which one is most likely to be accurate based on distilled historical insights.\n\n"
    )

    prompt += "### Distilled Experience from Similar Cases\n"
    prompt += "The following insights represent patterns, trends, and behaviors observed in highly similar historical time-series data:\n"

    if distilled_blocks:
        for i, block in enumerate(distilled_blocks, 1):
            prompt += f"[Insight {i}]: {block}\n"
    else:
        prompt += "No specific historical experience provided. Rely on standard time-series evaluation principles (e.g., trend consistency, seasonality, and noise levels).\n"

    prompt += "\n### Candidate Trajectories\n"
    prompt += "Compare the following predicted sequences:\n"
    for i, traj in enumerate(trajectories):
        prompt += f"Trajectory {i}: {_format_trajectory_values(traj)}\n"

    max_idx = len(trajectories) - 1
    prompt += (
        "\n### Instructions for Selection:\n"
        "1. Analyze each candidate trajectory against the provided [Insights].\n"
        "2. Evaluate which trajectory best aligns with the historical trends or characteristics described.\n"
        "3. Select the index of the single best trajectory.\n\n"
        "Final Requirement:\n"
        f"Return ONLY the index of the best trajectory using this exact format: <best>k</best>, where k is an integer between 0 and {max_idx}.\n"
        "Do not provide any explanation or additional text."
    )

    return prompt


def _parse_best_index(answer_text, num_candidates):
    if not answer_text:
        return None
    match = re.search(r"<best>\s*(\d+)\s*</best>", str(answer_text), re.IGNORECASE)
    if match:
        idx = int(match.group(1))
        if 0 <= idx < num_candidates:
            return idx
    fallback = re.findall(r"\b\d+\b", str(answer_text))
    for token in fallback:
        idx = int(token)
        if 0 <= idx < num_candidates:
            return idx
    return None


def select_best_trajectory_with_llm(current_series, trajectories, reasonings, data_name, attr, look_back, pred_window, top_k=3):
    historical_examples = _build_historical_examples(data_name, attr, look_back, pred_window)
    similar_examples = retrieve_similar_examples(current_series, historical_examples, top_k=top_k)
    distilled_blocks = []
    similar_meta = []
    for ex in similar_examples:
        idx = ex.get("index")
        similarity = ex.get("similarity")
        try:
            similarity_val = float(similarity) if similarity is not None else None
        except Exception:
            similarity_val = None
        similar_meta.append({"index": idx, "similarity": similarity_val})
        summary = _load_summary(data_name, attr, look_back, pred_window, idx)
        blocks = _extract_distilled_blocks(summary)
        if blocks:
            distilled_blocks.extend(blocks[:2])
    prompt = _build_llm_selection_prompt(distilled_blocks, trajectories)
    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Skipping LLM selection.")
        return None, similar_meta, None
    model = gpt4_api_output(api_key=api_key, temperature=0.2, top_p=0.7)
    try:
        _, answer = model(prompt)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None, similar_meta, None
    best_idx = _parse_best_index(answer, len(trajectories))
    if best_idx is None:
        print(f"Warning: Failed to parse best index from LLM answer. Answer excerpt: {str(answer)[:100]}...")
    return best_idx, similar_meta, answer


def compute_dtw(s1, s2):
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return np.inf
    dtw_matrix = np.full((n + 1, m + 1), np.inf, dtype=float)
    dtw_matrix[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(float(s1[i - 1]) - float(s2[j - 1]))
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
    return float(dtw_matrix[n, m])


def _safe_pearson_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return None
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 2:
        return None
    corr = np.corrcoef(a[mask], b[mask])[0, 1]
    if np.isnan(corr):
        return None
    return float(corr)


def _best_lag_by_cross_corr(target_hist, cov_hist, max_lag):
    max_lag = max(0, int(max_lag))
    x = np.asarray(target_hist, dtype=float)
    y = np.asarray(cov_hist, dtype=float)
    n = min(len(x), len(y))
    if n < 2:
        return None
    x = x[-n:]
    y = y[-n:]
    best_abs = -1.0
    best = None
    for lag in range(0, max_lag + 1):
        if lag == 0:
            x_seg = x
            y_seg = y
        else:
            x_seg = x[lag:]
            y_seg = y[:-lag]
        corr = _safe_pearson_corr(x_seg, y_seg)
        if corr is None:
            continue
        abs_corr = abs(corr)
        if abs_corr > best_abs:
            best_abs = abs_corr
            best = (int(lag), float(corr))
    return best


def _lagged_corr_score(target_hist, forecast, cov_hist_df, cov_future_df, period):
    if cov_hist_df is None or cov_future_df is None:
        return 0.0, None, None, None, None
    if not hasattr(cov_hist_df, "columns") or not hasattr(cov_future_df, "columns"):
        return 0.0, None, None, None, None
    common_cols = [c for c in cov_hist_df.columns if c in cov_future_df.columns]
    if len(common_cols) == 0:
        return 0.0, None, None, None, None
    max_lag = max(1, int(period))
    best_abs = -1.0
    best_col = None
    best_lag = None
    best_hist_corr = None
    for col in common_cols:
        best = _best_lag_by_cross_corr(target_hist, cov_hist_df[col].values, max_lag=max_lag)
        if best is None:
            continue
        lag, corr = best
        if abs(corr) > best_abs:
            best_abs = abs(corr)
            best_col = col
            best_lag = lag
            best_hist_corr = corr
    if best_col is None or best_lag is None:
        return 0.0, None, None, None, None
    hist_tail = np.asarray(cov_hist_df[best_col].values, dtype=float)[-max_lag:]
    fut = np.asarray(cov_future_df[best_col].values, dtype=float)
    cov_context = np.concatenate([hist_tail, fut], axis=0)
    fc = np.asarray(forecast, dtype=float)
    idx = (max_lag + np.arange(len(fc)) - int(best_lag)).astype(int)
    in_bounds = (idx >= 0) & (idx < len(cov_context))
    cov_aligned = np.full(len(fc), np.nan, dtype=float)
    if np.any(in_bounds):
        cov_aligned[in_bounds] = cov_context[idx[in_bounds]]
    valid = in_bounds & np.isfinite(fc) & np.isfinite(cov_aligned)
    if np.sum(valid) < 2:
        return 0.0, best_col, int(best_lag), best_hist_corr, None
    corr_future = _safe_pearson_corr(fc[valid], cov_aligned[valid])
    if corr_future is None:
        return 0.0, best_col, int(best_lag), best_hist_corr, None
    return float(abs(corr_future)), best_col, int(best_lag), best_hist_corr, float(corr_future)


def score_trajectory(history, forecast, period=24, cov_hist=None, cov_future=None):
    if len(forecast) == 0 or len(history) == 0:
        return {
            "total_score": 0.0,
            "jump_score": 0.0,
            "sigma_score": 0.0,
            "dtw_score": 0.0,
            "lag_score": 0.0,
            "lag_cov": None,
            "lag": None,
            "lag_hist_corr": None,
            "lag_future_corr": None,
        }
    hist = np.asarray(history, dtype=float)
    fc = np.asarray(forecast, dtype=float)
    if not np.isfinite(fc).any():
        return {
            "total_score": 0.0,
            "jump_score": 0.0,
            "sigma_score": 0.0,
            "dtw_score": 0.0,
            "lag_score": 0.0,
            "lag_cov": None,
            "lag": None,
            "lag_hist_corr": None,
            "lag_future_corr": None,
        }
    last_val = float(hist[-1])
    first_pred = float(fc[0])
    base_seg = hist
    hist_diff_std = float(np.std(np.diff(base_seg))) + 1e-6 if len(base_seg) > 1 else 1e-6
    jump_dist = abs(first_pred - last_val)
    jump_score = float(np.exp(-jump_dist / (2.0 * hist_diff_std)))

    local_window = hist
    mu = float(np.mean(local_window)) if len(local_window) > 0 else 0.0
    sigma = float(np.std(local_window)) + 1e-6 if len(local_window) > 0 else 1e-6
    outliers = np.logical_or(fc > mu + 3.0 * sigma, fc < mu - 3.0 * sigma)
    outlier_ratio = float(np.mean(outliers)) if len(fc) > 0 else 1.0
    sigma_score = 1.0 - outlier_ratio

    template = hist[-period:]
    if period == 0:
        dtw_score = 0.0
    else:
        dist = compute_dtw(fc[: len(template)], template)
        dtw_score = float(np.exp(-dist / (len(fc) * sigma)))
    lag_score, lag_cov, lag_value, lag_hist_corr, lag_future_corr = _lagged_corr_score(hist, fc, cov_hist, cov_future, period=period)
    total_score = (jump_score * 0.3) + (sigma_score * 0.2) + (lag_score * 0.5)
    return {
        "total_score": float(total_score),
        "jump_score": float(jump_score),
        "sigma_score": float(sigma_score),
        "dtw_score": float(dtw_score),
        "lag_score": float(lag_score),
        "lag_cov": lag_cov,
        "lag": lag_value,
        "lag_hist_corr": float(lag_hist_corr) if lag_hist_corr is not None else None,
        "lag_future_corr": float(lag_future_corr) if lag_future_corr is not None else None,
    }


def load_ground_truth(data_name, attr, look_back, pred_window, number):
    """
    加载指定样本的历史输入与真实预测数据 (EPF 版本，使用前80%数据)

    Args:
        data_name: 数据集名称 (如 'NP')
        attr: 属性名称
        look_back: 回看窗口大小
        pred_window: 预测窗口大小
        number: 样本序号 (滑动窗口局部序号)

    Returns:
        history_dates, history_values, future_dates, future_values,
        history_data, future_data, target_col, weather_hist, weather_future
    """
    data_dir = f'./datasets/{data_name}.csv'
    data = pd.read_csv(data_dir)

    # EPF：使用前 80% 数据作为评估范围，从 (80% - look_back) 处开始
    start_idx = int(len(data) * 0.8) - look_back
    if start_idx < 0:
        start_idx = 0
    data = data[start_idx:]

    slide_window = 24
    history_start_idx = number * slide_window
    history_end_idx = history_start_idx + look_back
    if history_start_idx < 0:
        history_start_idx = 0
    if history_end_idx > len(data):
        history_end_idx = len(data)
        history_start_idx = max(0, history_end_idx - look_back)

    future_start_idx = number * slide_window + look_back
    future_end_idx = future_start_idx + pred_window

    # 目标列模糊匹配
    target_col = attr
    if target_col not in data.columns:
        candidates = [
            col for col in data.columns
            if attr.lower().strip() == col.lower().strip()
            or attr.lower().strip() in col.lower().strip()
            or col.lower().strip() in attr.lower().strip()
        ]
        if candidates:
            target_col = candidates[0]

    history_data = data.iloc[history_start_idx:history_end_idx]
    future_data = data.iloc[future_start_idx:future_end_idx]

    history_dates = history_data['date'].tolist()
    history_values = history_data[target_col].tolist() if target_col in history_data else []

    future_dates = future_data['date'].tolist()
    future_values = future_data[target_col].tolist() if target_col in future_data else []

    weather_hist = None
    weather_future = None
    weather_path = f'./datasets/{data_name}_weather.csv'
    if os.path.exists(weather_path):
        weather_df = pd.read_csv(weather_path)
        if 'date' in weather_df.columns:
            weather_df = weather_df.set_index('date')
            weather_hist = weather_df.reindex(history_dates)
            weather_future = weather_df.reindex(future_dates)

    return history_dates, history_values, future_dates, future_values, history_data, future_data, target_col, weather_hist, weather_future


def evaluate_single_result(result_file, data_name, attr, look_back, pred_window, i):
    """
    评估单个结果文件，对文件内的多条轨迹打分：
      - 启发式分数 > 0.8 时直接选最优
      - 否则调用 LLM（基于 Memory_full distilled experience）辅助选择

    Args:
        result_file: 结果JSON文件路径
        data_name: 数据集名称
        attr: 属性名称
        look_back: 回看窗口大小
        pred_window: 预测窗口大小
        i: 全局段索引 (segment_index，如 476)

    Returns:
        评估结果列表
    """
    with open(result_file, 'r') as f:
        results = json.load(f)

    evaluation_results = []

    result_dir = os.path.dirname(result_file)
    output_root = os.path.abspath(os.path.join(result_dir, os.pardir, os.pardir))
    figures_root = os.path.join(output_root, 'figures', data_name, attr)
    os.makedirs(figures_root, exist_ok=True)
    segment_id = f'{i}'

    # 1. 收集所有预测序列
    all_pred_values = []
    all_reasonings = []
    valid_sample_indices = []

    for item in results:
        index = item['index']
        answer = item.get('answer', '')
        reasoning = item.get('reasoning', '')

        if isinstance(answer, (dict, list)):
            answer_text = str(answer)
        elif answer is None:
            answer_text = ""
        else:
            answer_text = str(answer)

        try:
            pred_dates, pred_values = parse_prediction_from_answer(answer_text)
        except Exception as e:
            print(f"Error parsing answer for index {index}: {e}")
            pred_dates, pred_values = [], []

        numeric_values = []
        for val in pred_values:
            try:
                numeric_values.append(float(val))
            except (ValueError, TypeError):
                numeric_values.append(np.nan)

        if len(numeric_values) > 0:
            all_pred_values.append(numeric_values)
            all_reasonings.append(reasoning)
            valid_sample_indices.append(index)

    if not all_pred_values:
        print(f"⚠️  Warning: Segment {i} - No valid predictions found.")
        return []

    # 2. 加载真实值（EPF 偏移：局部序号 = i - 476）
    history_dates, history_values, true_dates, true_values, history_df, future_df, target_col, weather_hist, weather_future = load_ground_truth(
        data_name, attr, look_back, pred_window, i - 476
    )

    true_numeric = []
    for val in true_values:
        try:
            true_numeric.append(float(val))
        except (ValueError, TypeError):
            true_numeric.append(np.nan)

    # 3. 对齐长度
    min_len = len(true_numeric)
    for p in all_pred_values:
        if len(p) < min_len:
            min_len = len(p)

    if min_len == 0:
        print(f"⚠️  Warning: Segment {i} - Effective length is 0.")
        return []

    aligned_preds = [p[:min_len] for p in all_pred_values]
    aligned_true = true_numeric[:min_len]

    hist_numeric = []
    for value in history_values:
        try:
            hist_numeric.append(float(value))
        except (TypeError, ValueError):
            hist_numeric.append(np.nan)
    hist_array = np.array(hist_numeric, dtype=float)
    if len(hist_array) == 0:
        print(f"⚠️  Warning: Segment {i} - empty history values")
        return []

    if weather_hist is not None and weather_future is not None and 'tmax' in weather_hist.columns and 'tmax' in weather_future.columns:
        cov_hist = weather_hist[['tmax']]
        cov_future = weather_future[['tmax']]
    else:
        cov_hist = None
        cov_future = None

    # 4. 启发式打分
    scores = []
    trajectory_details = []
    for p in aligned_preds:
        p_arr = np.array(p, dtype=float)
        if not np.isfinite(p_arr).any():
            scores.append(-np.inf)
            trajectory_details.append({
                "total_score": float("-inf"),
                "jump_score": 0.0,
                "sigma_score": 0.0,
                "dtw_score": 0.0,
                "lag_score": 0.0,
                "lag_cov": None,
                "lag": None,
                "lag_hist_corr": None,
                "lag_future_corr": None,
            })
            continue
        detail = score_trajectory(hist_array, p_arr, period=24, cov_hist=cov_hist, cov_future=cov_future)
        scores.append(detail["total_score"])
        trajectory_details.append(detail)

    if len(scores) == 0 or np.all(np.isneginf(scores)):
        print(f"⚠️  Warning: Segment {i} - no valid scores")
        return []

    # 5. 选择策略：启发式分数 > 0.8 直接选，否则调用 LLM
    max_heuristic_score = np.nanmax(scores)
    llm_best_idx = None
    llm_similar_examples = []
    llm_answer = None

    if max_heuristic_score > 0.8:
        print(f"  Segment {i}: Max heuristic score {max_heuristic_score:.4f} > 0.8. Using heuristic selection.")
        best_idx = int(np.nanargmax(scores))
        selection_method = "heuristic"
    else:
        llm_best_idx, llm_similar_examples, llm_answer = select_best_trajectory_with_llm(
            hist_array,
            aligned_preds,
            all_reasonings,
            data_name,
            attr,
            look_back,
            pred_window,
            top_k=3
        )
        if llm_best_idx is not None and 0 <= llm_best_idx < len(aligned_preds):
            best_idx = int(llm_best_idx)
            selection_method = "llm"
        else:
            best_idx = int(np.nanargmax(scores))
            selection_method = "heuristic"

    best_pred_full = np.array(aligned_preds[best_idx], dtype=float)
    metric_mask = np.isfinite(aligned_true) & np.isfinite(best_pred_full)
    if not np.any(metric_mask):
        print(f"⚠️  Warning: Segment {i} - no finite values for metric computation")
        return []

    final_true = np.array(aligned_true, dtype=float)[metric_mask]
    final_pred = best_pred_full[metric_mask]

    mse = MSE(final_true, final_pred)
    mae = MAE(final_true, final_pred)
    rmse = np.sqrt(mse)

    print(f"  Segment {i} trajectory scores (selection={selection_method}):")
    for k, (sample_index, detail) in enumerate(zip(valid_sample_indices, trajectory_details)):
        score_val = detail.get("total_score")
        jump_val = detail.get("jump_score")
        sigma_val = detail.get("sigma_score")
        dtw_val = detail.get("dtw_score")
        lag_val = detail.get("lag_score")
        lag_cov = detail.get("lag_cov")
        lag_num = detail.get("lag")
        flag = " <== selected" if k == best_idx else ""
        print(f"    Trajectory {k} (index {sample_index}): score={score_val:.6f} jump={jump_val:.4f} sigma={sigma_val:.4f} dtw={dtw_val:.4f} lag={lag_val:.4f} cov={lag_cov} lag={lag_num}{flag}")

    # 6. 绘图
    fig, ax = plt.subplots(figsize=(10, 4.8))
    hist_len = len(hist_array)
    fut_len = len(aligned_true)

    x_hist = np.arange(-hist_len, 0, 1)
    x_future = np.arange(0, fut_len, 1)

    if hist_len > 0:
        ax.plot(x_hist, hist_array, color='#2ca02c', linewidth=1.4, label='History Input')

    ax.plot(x_future, aligned_true, label='GroundTruth', color='#1f77b4', linewidth=1.4)
    ax.plot(x_future, best_pred_full, label='Best Prediction', color='#ff7f0e', linewidth=1.8)

    for p in aligned_preds:
        ax.plot(x_future, p, color='#ff7f0e', alpha=0.1, linewidth=0.5)

    forecast_marker = -0.5 if hist_len > 0 else -0.1
    ax.axvline(forecast_marker, color='#888', linestyle='--', linewidth=1.1)

    selected_sample_index = valid_sample_indices[best_idx] if best_idx < len(valid_sample_indices) else None
    selected_score = float(scores[best_idx]) if best_idx < len(scores) else None
    title = (
        f'{attr} Best Prediction · Segment {segment_id} · Samples: {len(valid_sample_indices)} · Score: {selected_score:.4f}'
        if selected_score is not None
        else f'{attr} Best Prediction · Segment {segment_id} · Samples: {len(valid_sample_indices)}'
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Time Index')
    ax.set_ylabel(attr)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    figure_path = os.path.join(figures_root, f'{segment_id}_best.png')
    fig.savefig(figure_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

    full_true = np.concatenate([hist_array, aligned_true])
    full_pred = np.concatenate([hist_array, best_pred_full])
    pdf_path = os.path.join(figures_root, f'{segment_id}_best.pdf')
    try:
        visual(full_true.tolist(), full_pred.tolist(), pdf_path)
    except Exception as exc:
        pdf_path = None

    figure_rel_path = os.path.relpath(figure_path, start=output_root)
    pdf_rel_path = os.path.relpath(pdf_path, start=output_root) if pdf_path else None

    result_entry = {
        'index': i,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'pred_length': len(best_pred_full),
        'true_length': len(true_values),
        'length_diff': len(best_pred_full) - len(true_values),
        'length_status': "LLM Selected" if selection_method == "llm" else "Best Selected",
        'figure_path': figure_rel_path,
        'figure_pdf_path': pdf_rel_path,
        'num_aggregated_samples': len(valid_sample_indices),
        'num_candidates': len(valid_sample_indices),
        'selected_sample_index': selected_sample_index,
        'selected_score': selected_score,
        'selection_method': selection_method,
        'llm_similar_examples': llm_similar_examples,
        'llm_answer': llm_answer,
        'trajectory_scores': [
            {
                'sample_index': int(sample_index) if sample_index is not None else None,
                'total_score': float(detail.get('total_score')) if detail.get('total_score') is not None else None,
                'jump_score': float(detail.get('jump_score')) if detail.get('jump_score') is not None else None,
                'sigma_score': float(detail.get('sigma_score')) if detail.get('sigma_score') is not None else None,
                'dtw_score': float(detail.get('dtw_score')) if detail.get('dtw_score') is not None else None,
                'lag_score': float(detail.get('lag_score')) if detail.get('lag_score') is not None else None,
                'lag_cov': detail.get('lag_cov'),
                'lag': int(detail.get('lag')) if detail.get('lag') is not None else None,
                'lag_hist_corr': float(detail.get('lag_hist_corr')) if detail.get('lag_hist_corr') is not None else None,
                'lag_future_corr': float(detail.get('lag_future_corr')) if detail.get('lag_future_corr') is not None else None,
                'selected': bool(k == best_idx),
            }
            for k, (sample_index, detail) in enumerate(zip(valid_sample_indices, trajectory_details))
        ]
    }

    evaluation_results.append(result_entry)
    return evaluation_results


def evaluate_all_results(result_dir, data_name, look_back=168, pred_window=24):
    """
    评估所有结果文件

    Args:
        result_dir: 结果目录路径
        data_name: 数据集名称
        look_back: 回看窗口大小
        pred_window: 预测窗口大小

    Returns:
        所有评估结果的汇总
    """
    attrs = ['OT']

    all_evaluations = {}

    for attr in attrs:
        print(f"\n{'='*60}")
        print(f"Evaluating {attr} - Collecting all segment results...")
        print(f"{'='*60}")

        attr_all_results = []

        for i in range(30):
            segment_index = i + 476
            result_file = os.path.join(result_dir, f'result_{attr}_{data_name}_{look_back}_{pred_window}_{segment_index}.json')

            if os.path.exists(result_file):
                print(f"  Processing segment {segment_index}...")
                eval_results = evaluate_single_result(result_file, data_name, attr, look_back, pred_window, segment_index)
                attr_all_results.extend(eval_results)
            else:
                print(f"  Segment {segment_index} not found, skipping...")

        valid_results = [r for r in attr_all_results if r['mse'] is not None]

        if valid_results:
            avg_mse = np.mean([r['mse'] for r in valid_results])
            avg_mae = np.mean([r['mae'] for r in valid_results])
            avg_rmse = np.mean([r['rmse'] for r in valid_results])

            all_evaluations[attr] = {
                'individual_results': attr_all_results,
                'average_mse': avg_mse,
                'average_mae': avg_mae,
                'average_rmse': avg_rmse,
                'num_samples': len(valid_results),
                'total_segments': len(attr_all_results)
            }

            print(f"\n📊 Final Results for {attr}:")
            print(f"  Average MSE:  {avg_mse:.6f}")
            print(f"  Average MAE:  {avg_mae:.6f}")
            print(f"  Average RMSE: {avg_rmse:.6f}")
            print(f"  Valid samples: {len(valid_results)}/{len(attr_all_results)}")
            print(f"  Available segments: {len(attr_all_results)}/30")

            print(f"\n  Individual samples:")
            for r in attr_all_results:
                if r['mse'] is not None:
                    length_info = f"[{r['length_status']}]" if r['length_status'] not in ("匹配", "Best Selected") else ""
                    print(f"    Index {r['index']}: MSE={r['mse']:.6f}, MAE={r['mae']:.6f}, RMSE={r['rmse']:.6f} [{r['selection_method']}]{length_info}")
                else:
                    print(f"    Index {r['index']}: No valid evaluation")
        else:
            print(f"⚠️  No valid results for {attr}")
            all_evaluations[attr] = None

    return all_evaluations


def print_summary(all_evaluations):
    """
    打印评估结果摘要
    """
    print(f"\n{'='*60}")
    print("📈 EVALUATION SUMMARY")
    print(f"{'='*60}\n")

    print(f"{'Attribute':<10} {'Avg MSE':<15} {'Avg MAE':<15} {'Avg RMSE':<15} {'Samples':<10}")
    print("-" * 65)

    for attr, results in all_evaluations.items():
        if results is not None:
            print(f"{attr:<10} {results['average_mse']:<15.6f} {results['average_mae']:<15.6f} "
                  f"{results['average_rmse']:<15.6f} {results['num_samples']:<10}")
        else:
            print(f"{attr:<10} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<10}")

    print("-" * 65)

    valid_attrs = [k for k, v in all_evaluations.items() if v is not None]
    if valid_attrs:
        overall_mse = np.mean([all_evaluations[k]['average_mse'] for k in valid_attrs])
        overall_mae = np.mean([all_evaluations[k]['average_mae'] for k in valid_attrs])
        overall_rmse = np.mean([all_evaluations[k]['average_rmse'] for k in valid_attrs])

        print(f"{'Overall':<10} {overall_mse:<15.6f} {overall_mae:<15.6f} {overall_rmse:<15.6f}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    data_name = 'NP'
    result_dir = f'./results/EPF/result_uq/{data_name}'
    output_file = f'./results/EPF/result_uq/{data_name}_evaluation_results_uqmem.json'
    print(result_dir)
    look_back = 168
    pred_window = 24

    all_evaluations = evaluate_all_results(result_dir, data_name, look_back, pred_window)
    print_summary(all_evaluations)

    with open(output_file, 'w') as f:
        serializable_results = {}
        for attr, results in all_evaluations.items():
            if results is not None:
                serializable_results[attr] = {
                    'average_mse': float(results['average_mse']),
                    'average_mae': float(results['average_mae']),
                    'average_rmse': float(results['average_rmse']),
                    'num_samples': int(results['num_samples']),
                    'individual_results': [
                        {
                            'index': int(r['index']),
                            'mse': float(r['mse']) if r['mse'] is not None else None,
                            'mae': float(r['mae']) if r['mae'] is not None else None,
                            'rmse': float(r['rmse']) if r['rmse'] is not None else None,
                            'pred_length': int(r['pred_length']),
                            'true_length': int(r['true_length']),
                            'length_diff': int(r['length_diff']),
                            'length_status': r['length_status'],
                            'num_candidates': int(r.get('num_candidates')) if r.get('num_candidates') is not None else None,
                            'selected_sample_index': int(r.get('selected_sample_index')) if r.get('selected_sample_index') is not None else None,
                            'selected_score': float(r.get('selected_score')) if r.get('selected_score') is not None else None,
                            'selection_method': r.get('selection_method'),
                            'llm_similar_examples': r.get('llm_similar_examples'),
                            'llm_answer': r.get('llm_answer'),
                            'trajectory_scores': r.get('trajectory_scores'),
                            'figure_path': r.get('figure_path'),
                            'figure_pdf_path': r.get('figure_pdf_path')
                        }
                        for r in results['individual_results']
                    ]
                }
            else:
                serializable_results[attr] = None

        json.dump(serializable_results, f, indent=2)

    print(f"✅ Evaluation results saved to: {output_file}")
