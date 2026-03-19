import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import json
import time
import os
import re
import random
import warnings
import shutil
from utils.api_ouput import deepseek_api_output
from utils.api_ouput import gpt4_api_output
from utils.metrics import MSE, MAE
from main.ETTh.ETTh_main_summary import analyze_memory_reasoning

debug=0
REFLECT=0

meaning_dict = {
    'HUFL': 'High Useful Load (high-frequency component of useful electrical load)',
    'HULL': 'High UnUseful Load (high-frequency component of non-useful electrical load)',
    'MUFL': 'Medium Useful Load (medium-frequency component of useful electrical load)',
    'MULL': 'Medium UnUseful Load (medium-frequency component of non-useful electrical load)',
    'LUFL': 'Low Useful Load (low-frequency component of useful electrical load)',
    'LULL': 'Low UnUseful Load (low-frequency component of non-useful electrical load)',
    'OT': 'Oil Temperature (transformer oil temperature, prediction target)',
    'tmax': 'Daily maximum temperature',
    'tmin': 'Daily minimum temperature'
}


def extract_time_series_features(series):
    """
    提取时间序列特征用于相似性计算
    """
    # 检查输入数据是否有效
    if len(series) == 0:
        # 返回默认特征向量
        return np.array([0.0] * 12)
    
    features = []
    
    # 统计特征
    features.extend([
        np.mean(series),
        np.std(series) if len(series) > 1 else 0.0,
        np.min(series),
        np.max(series),
        np.median(series)
    ])
    
    # 趋势特征
    if len(series) > 1:
        try:
            trend_slope = np.polyfit(range(len(series)), series, 1)[0]
            features.append(trend_slope)
        except:
            features.append(0.0)
    else:
        features.append(0.0)
    
    # 周期性特征 (假设24小时周期)
    if len(series) >= 24:
        daily_std = []
        for i in range(0, len(series), 24):
            if i + 24 <= len(series):
                daily_std.append(np.std(series[i:i+24]))
        features.append(np.mean(daily_std) if daily_std else 0.0)
    else:
        features.append(0.0)
    
    # 自相关特征
    if len(series) > 1:
        try:
            lag1_corr = np.corrcoef(series[:-1], series[1:])[0, 1]
            features.append(lag1_corr if not np.isnan(lag1_corr) else 0.0)
        except:
            features.append(0.0)
    else:
        features.append(0.0)
    
    # 变化率特征
    if len(series) > 1:
        try:
            changes = np.diff(series)
            features.extend([
                np.mean(np.abs(changes)),
                np.std(changes) if len(changes) > 1 else 0.0
            ])
        except:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0])
    
    return np.array(features)


def _summary_dir_for(data_name):
    return os.path.join("Memory", "cases", "ETTh", "summary_outputs", data_name)


def _summary_path(data_name, attr, look_back, pred_window, index):
    return os.path.join(
        _summary_dir_for(data_name),
        f"result_{attr}_{data_name}_{look_back}_{pred_window}_{index}.json"
    )


def _load_summary(data_name, attr, look_back, pred_window, index):
    path = _summary_path(data_name, attr, look_back, pred_window, index)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


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
    
    # --- Role and Task Context ---
    prompt = (
        "Role: You are a world-class Expert in Time Series Forecasting and Evaluation.\n"
        "Task: Your objective is to analyze multiple candidate trajectories (predictions) "
        "and determine which one is most likely to be accurate based on distilled historical insights.\n\n"
    )

    # --- Knowledge Injection: Distilled Experience ---
    prompt += "### Distilled Experience from Similar Cases\n"
    prompt += "The following insights represent patterns, trends, and behaviors observed in highly similar historical time-series data:\n"
    
    if distilled_blocks:
        for i, block in enumerate(distilled_blocks, 1):
            prompt += f"[Insight {i}]: {block}\n"
    else:
        prompt += "No specific historical experience provided. Rely on standard time-series evaluation principles (e.g., trend consistency, seasonality, and noise levels).\n"

    # --- Input Data: Candidate Trajectories ---
    prompt += "\n### Candidate Trajectories\n"
    prompt += "Compare the following predicted sequences:\n"
    for i, traj in enumerate(trajectories):
        # Assuming _format_trajectory_values is defined elsewhere
        prompt += f"Trajectory {i}: {_format_trajectory_values(traj)}\n"

    # --- Decision Criteria and Output Format ---
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


def select_best_trajectory_with_llm(current_values, historical_examples, trajectories, api_key, data_name, attr, look_back, pred_window):
    similar_examples = retrieve_similar_examples(current_values, historical_examples, top_k=3)
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
    model = deepseek_api_output(api_key=api_key, temperature=0.6, top_p=0.7)
    try:
        _, answer = model(prompt)
    except Exception:
        return None, similar_meta, None
    best_idx = _parse_best_index(answer, len(trajectories))
    return best_idx, similar_meta, answer

def _extract_feature_stats(series):
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}

    feats = extract_time_series_features(arr).tolist()
    feature_names = [
        "mean",
        "std",
        "min",
        "max",
        "median",
        "trend_slope",
        "daily_std_24h",
        "lag1_corr",
        "mean_abs_change",
        "std_change",
    ]

    out = {}
    for idx, val in enumerate(feats):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        try:
            fval = float(val)
        except Exception:
            fval = 0.0
        if not np.isfinite(fval):
            fval = 0.0
        out[name] = fval
    return out

def _format_feature_stats(title, series):
    stats = _extract_feature_stats(series)
    return f"- {title}: {json.dumps(stats, ensure_ascii=False)}"


def get_result(text):
    """
    从模型输出中提取数值结果
    """
    res_list = []
    pattern = r"```([\s\S]*?)```"
    match = re.search(pattern, text)
    if match is None:
        pattern = r"\|.*\|" 
        text_list = re.findall(pattern, text)
    else:
        text = match.group(1).strip()
        text_list = text.split('\n')
    
    for _, item in enumerate(text_list):
        item = item.strip()
        if item.endswith('|'):
            item = item.rstrip('|').strip()
        match = re.search(r'(-?\d+\.\d+|-?\d+)$', item)
        if match:
            number = float(match.group(0))
            res_list.append(number)
    return res_list


def _build_reflection_block(feedback_text):
    if not feedback_text:
        return ""

    feedback = feedback_text.strip()
    if not feedback:
        return ""

    prefix = "QC failed:"
    if feedback.lower().startswith(prefix.lower()):
        feedback = feedback[len(prefix):].strip()

    issues = [
        issue.strip()
        for issue in re.split(r";|\n", feedback)
        if issue.strip()
    ]

    hard_keywords = [
        "expected",
        "format",
        "missing",
        "non-numeric",
        "duplicate",
        "must",
        "complete",
        "timestamp",
        "value",
        "empty",
        "boundary jump",
        "out of range"
    ]

    hard_constraints = []
    soft_constraints = []

    for issue in issues:
        normalized_issue = issue.rstrip(".").strip()
        if not normalized_issue:
            continue
        constraint = f"Must address: {normalized_issue}."
        if any(keyword in normalized_issue.lower() for keyword in hard_keywords):
            hard_constraints.append(constraint)
        else:
            soft_constraints.append(constraint)

    block_lines = [
        "\n\n**Reflection / Corrections:**",
        f"Last QC feedback: {feedback}"
    ]

    if hard_constraints:
        block_lines.append("Hard constraints:")
        block_lines.extend(f"- {item}" for item in hard_constraints)
    if soft_constraints:
        block_lines.append("Soft constraints:")
        block_lines.extend(f"- {item}" for item in soft_constraints)

    return "\n".join(block_lines) + "\n"

def _extract_forecast_lines(answer_text):
    """
    Extract forecast lines from the model answer.

    Returns:
        tuple[list[str] | None, str]: (lines, error_message)
    """
    if not answer_text or not answer_text.strip():
        return None, "Answer text is empty."

    # 优先匹配标准格式
    match = re.search(r"<answer>\s*```(.*?)```\s*</answer>", answer_text, re.DOTALL | re.IGNORECASE)

    # 若未匹配成功，再尝试宽松格式
    if not match:
        match = re.search(r"<answer>(.*?)</answer>", answer_text, re.DOTALL | re.IGNORECASE)

    if not match:
        return None, "Answer does not contain a recognizable <answer> ... </answer> block."


    # match = re.search(r"<answer>\s*```(.*?)```\s*</answer>", answer_text, re.DOTALL | re.IGNORECASE)
    # if not match:
    #     return None, "Answer does not follow the required <answer> ```...``` </answer> format."

    block = match.group(1).strip()
    if not block:
        return None, "Forecast block inside triple backticks is empty."

    raw_lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not raw_lines:
        return None, "Forecast block does not contain any data lines."

    lines = [line for line in raw_lines if re.search(r'\d', line)]
    if not lines:
        return None, "Forecast block contains no timestamp/value rows after removing headers."

    return lines, ""


def _quick_quality_check(answer_text, pred_window, historical_values, volatility_ratio_threshold=1.3):
    """
    Perform fast quality checks on the model answer.

    Returns:
        tuple[bool, str, list[dict], dict]: (passed, feedback, parsed_forecast, metrics)
    """
    lines, error = _extract_forecast_lines(answer_text)
    if lines is None:
        return False, error, [], {}

    issues = []
    parsed = []

    if len(lines) < pred_window:
        issues.append(f"Forecast contains only {len(lines)} lines; requires at least {pred_window}.")

    extra_lines = max(0, len(lines) - pred_window)
    lines_to_process = lines[:pred_window]

    timestamps = set()

    for idx, line in enumerate(lines_to_process, start=1):
        if ',' in line:
            parts = line.split(',', 1)
            if len(parts) != 2:
                issues.append(f"Line {idx} is not in 'timestamp, value' format: '{line}'.")
                continue
            timestamp, value_str = parts[0].strip(), parts[1].strip()
        else:
            parts = line.rsplit(' ', 1)
            if len(parts) != 2:
                issues.append(f"Line {idx} is not in 'timestamp value' format: '{line}'.")
                continue
            timestamp, value_str = parts[0].strip(), parts[1].strip()
        if not timestamp:
            issues.append(f"Line {idx} missing timestamp before the value.")

        # Attempt to parse numeric value
        try:
            value = float(value_str.replace(',', ''))
        except ValueError:
            issues.append(f"Line {idx} has a non-numeric value: '{value_str}'.")
            continue

        if timestamp in timestamps:
            issues.append(f"Duplicate timestamp detected: '{timestamp}'.")
        else:
            timestamps.add(timestamp)

        parsed.append({'timestamp': timestamp, 'value': value})

    forecast_values = [item['value'] for item in parsed]

    hist_series = pd.Series(historical_values).astype(float).dropna().to_numpy()
    if hist_series.size >= 2:
        historical_diff_std = float(np.nanstd(np.diff(hist_series)))
    else:
        historical_diff_std = 0.0

    if len(forecast_values) >= 2:
        forecast_diff_std = float(np.nanstd(np.diff(forecast_values)))
    else:
        forecast_diff_std = 0.0

    metrics = {
        'historical_diff_std': historical_diff_std,
        'forecast_diff_std': forecast_diff_std
    }
    if extra_lines:
        metrics['extra_forecast_lines'] = extra_lines

    boundary_threshold = None
    if hist_series.size > 0 and forecast_values:
        boundary_diff = abs(forecast_values[0] - hist_series[-1])
        metrics['boundary_jump'] = boundary_diff

        if historical_diff_std <= 1e-6:
            boundary_threshold = 1e-3
        else:
            boundary_threshold = historical_diff_std * volatility_ratio_threshold

        metrics['boundary_jump_limit'] = boundary_threshold
        if boundary_diff > boundary_threshold:
            issues.append(
                (
                    "Boundary jump too large: |forecast[0] - history[-1]| "
                    f"= {boundary_diff:.4f} (limit {boundary_threshold:.4f})."
                )
            )

    if hist_series.size > 0 and forecast_values:
        hist_min = float(np.nanmin(hist_series))
        hist_max = float(np.nanmax(hist_series))
        hist_mean = float(np.nanmean(hist_series))
        hist_value_std = float(np.nanstd(hist_series))

        if hist_value_std <= 1e-6:
            sigma = max(1e-3, abs(hist_max - hist_min) * 0.1)
        else:
            sigma = hist_value_std

        range_lower = hist_mean - 3.0 * sigma
        range_upper = hist_mean + 3.0 * sigma

        metrics['allowed_value_range'] = [range_lower, range_upper]
        metrics['historical_value_mean'] = hist_mean
        metrics['historical_value_std'] = hist_value_std

        out_of_range_indices = [
            idx for idx, value in enumerate(forecast_values, start=1)
            if value < range_lower or value > range_upper
        ]

        if out_of_range_indices:
            issues.append(
                (
                    "Forecast values out of acceptable range for indices "
                    f"{out_of_range_indices} (allowed [{range_lower:.4f}, {range_upper:.4f}])."
                )
            )

    if historical_diff_std <= 1e-6:
        if forecast_diff_std > 1e-3:
            issues.append(
                f"Forecast volatility too high: diff std {forecast_diff_std:.4f} while historical diff std is near zero."
            )
    else:
        allowed_std = historical_diff_std * volatility_ratio_threshold
        metrics['allowed_forecast_diff_std'] = allowed_std
        if forecast_diff_std > allowed_std:
            issues.append(
                (
                    "Forecast volatility too high: diff std "
                    f"{forecast_diff_std:.4f} vs historical {historical_diff_std:.4f} "
                    f"(limit {allowed_std:.4f})."
                )
            )

    if issues:
        feedback = "QC failed: " + "; ".join(issues)
        return False, feedback, parsed, metrics

    return True, "QC passed: forecast format and length look good.", parsed, metrics


def _run_with_reflection(model, prompt, pred_window, historical_values):
    while True:
        try:
            reasoning, result = model(prompt)
            break
        except Exception as e:
            print(f"Error: {e}")
            reasoning = 'Error'
            result = 'Error'
            time.sleep(1)

    if int(REFLECT) != 1:
        return reasoning, result

    passed, feedback, _, _ = _quick_quality_check(result, pred_window, historical_values)
    if passed:
        return reasoning, result

    retry_prompt = prompt + _build_reflection_block(feedback)
    while True:
        try:
            retry_reasoning, retry_result = model(retry_prompt)
            return retry_reasoning, retry_result
        except Exception as e:
            print(f"Error: {e}")
            retry_reasoning = 'Error'
            retry_result = 'Error'
            time.sleep(1)


def retrieve_similar_examples(current_data, historical_examples, top_k=5):
    """
    基于时间序列特征相似性检索相似的样例
    """
    if not historical_examples:
        return []
    valid_examples = [
        ex for ex in historical_examples
        if isinstance(ex, dict)
        and ex.get("summary_file")
        and os.path.exists(ex.get("summary_file"))
    ]
    if not valid_examples:
        return []
    # 提取当前数据的特征
    current_features = extract_time_series_features(current_data)
    
    # 提取历史样例的特征
    historical_features = []
    for example in valid_examples:
        features = extract_time_series_features(example['data'])
        historical_features.append(features)
    
    historical_features = np.array(historical_features)
    
    # 标准化特征
    scaler = StandardScaler()
    scaler.fit(historical_features)
    current_features_scaled = scaler.transform(current_features.reshape(1, -1))
    historical_features_scaled = scaler.transform(historical_features)
    
    if debug:
        print(f"[DEBUG] current_features: {current_features}")
        print(f"[DEBUG] historical_features shape: {historical_features.shape}")
        print(f"[DEBUG] historical_features sample: {historical_features[:3]}")
        print(f"[DEBUG] scaler.mean_: {scaler.mean_}")
        print(f"[DEBUG] scaler.scale_: {scaler.scale_}")
        print(f"[DEBUG] any NaN in current_features_scaled: {np.isnan(current_features_scaled).any()}")
        print(f"[DEBUG] any NaN in historical_features_scaled: {np.isnan(historical_features_scaled).any()}")
        print(f"[DEBUG] current_features_scaled: {current_features_scaled}")
    
    # 计算余弦相似度
    similarities = cosine_similarity(current_features_scaled, historical_features_scaled)[0]
    if debug:
        print(f"[DEBUG] similarities: {similarities}")
    
    # 获取最相似的top_k个样例
    combined_scores = []
    for idx, sim in enumerate(similarities):
        confidence = valid_examples[idx].get("confidence")
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        combined_scores.append(sim + confidence)
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    
    similar_examples = []
    for idx in top_indices:
        similar_examples.append({
            'example': valid_examples[idx],
            'similarity': similarities[idx],
            'confidence': valid_examples[idx].get("confidence", 0),
            'combined_score': combined_scores[idx],
            'index': valid_examples[idx].get('index', idx)
        })
    
    return similar_examples


def compute_dtw(s1, s2):
    """
    计算两个序列之间的DTW距离
    """
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # insertion
                                          dtw_matrix[i, j - 1],    # deletion
                                          dtw_matrix[i - 1, j - 1]) # match
    return dtw_matrix[n, m]


def retrieve_similar_examples_dtw(current_data, historical_examples, top_k=5):
    """
    基于DTW距离检索相似的样例
    """
    if not historical_examples:
        return []
    valid_examples = [
        ex for ex in historical_examples
        if isinstance(ex, dict)
        and ex.get("summary_file")
        and os.path.exists(ex.get("summary_file"))
    ]
    if not valid_examples:
        return []
    distances = []
    
    for idx, example in enumerate(valid_examples):
        hist_data = example['data']
        # 计算DTW距离
        dist = compute_dtw(current_data, hist_data)
        confidence = example.get("confidence")
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        dtw_score = np.exp(-dist)
        combined_score = dtw_score + confidence

        distances.append({
            'example': example,
            'similarity': dtw_score, # exp(-dist)，距离越小分数越接近1
            'confidence': confidence,
            'combined_score': combined_score,
            'distance': dist,
            'index': example.get('index', idx)
        })
    
    distances.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # 获取top_k
    similar_examples = distances[:top_k]
    
    return similar_examples



def evaluate_prediction_quality(predicted, actual):
    """
    评估预测质量，返回质量分数
    """
    if len(predicted) == 0 or len(actual) == 0:
        return 0.0
    
    # 确保长度一致
    min_len = min(len(predicted), len(actual))
    predicted = predicted[:min_len]
    actual = actual[:min_len]
    
    # 计算多个指标
    mae = MAE(predicted, actual)
    mse = MSE(predicted, actual)
    rmse = np.sqrt(mse)
    
    # 计算MAPE (避免除零)
    mape = 0.0
    if np.mean(np.abs(actual)) > 1e-8:
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # 综合质量分数 (越小越好，转换为越大越好)
    # 使用倒数并归一化
    quality_score = 1.0 / (1.0 + mae + rmse * 0.1 + mape * 0.01)
    
    return quality_score


def evaluate_reasoning_quality(reasoning_text):
    """
    评估推理质量，基于推理文本的特征
    """
    if not reasoning_text or reasoning_text == 'Error':
        return 0.0
    
    quality_score = 0.0
    
    # 推理长度 (适中的长度更好)
    reasoning_length = len(reasoning_text)
    if 100 <= reasoning_length <= 2000:
        quality_score += 0.3
    elif reasoning_length > 2000:
        quality_score += 0.2
    
    # 关键词检测
    reasoning_lower = reasoning_text.lower()
    quality_keywords = ['step', 'analyze', 'trend', 'pattern', 'forecast', 'reasoning', 'consider']
    keyword_count = sum(1 for keyword in quality_keywords if keyword in reasoning_lower)
    quality_score += min(keyword_count * 0.1, 0.4)
    
    # 结构化程度 (包含数字步骤)
    step_pattern = r'\d+\.|step \d+|first|second|third|finally'
    step_matches = len(re.findall(step_pattern, reasoning_lower))
    quality_score += min(step_matches * 0.05, 0.3)
    
    return min(quality_score, 1.0)


def rerank_examples_by_quality(similar_examples):
    """
    根据预测质量和推理质量对检索到的样例进行重新排序
    使用Memory中存储的训练集结果
    """
    reranked_examples = []
    
    for example_info in similar_examples:
        similarity = example_info['similarity']
        example = example_info['example']
        
        # 从Memory中获取推理结果
        forecast_string = example.get('forecast_string', '')
        reasoning_string = example.get('reasoning_string', '')
        
        # 评估预测质量
        predicted = get_result(forecast_string)
        # 基于预测结果的完整性评估质量
        predicted_quality = 0.8 if len(predicted) == 96 else 0.3  # 假设pred_window=96
        
        # 评估推理质量
        reasoning_quality = evaluate_reasoning_quality(reasoning_string)
        
        # 评估预测结果的合理性 (基于数值范围等)
        prediction_reasonableness = evaluate_prediction_reasonableness(predicted)
        
        if debug:
            print(f"[DEBUG] Example idx={example.get('index', -1)} similarity={similarity:.6f}")
            print(f"[DEBUG] forecast_string_present={bool(forecast_string.strip())} predicted_len={len(predicted)} predicted_quality={predicted_quality:.3f}")
            print(f"[DEBUG] reasoning_len={len(reasoning_string)} reasoning_quality={reasoning_quality:.3f}")
        if len(predicted) > 0:
            min_val = min(predicted)
            max_val = max(predicted)
            diffs = [abs(predicted[i] - predicted[i-1]) for i in range(1, len(predicted))]
            avg_diff = sum(diffs)/len(diffs) if diffs else 0.0
            unique_ratio = len(set(predicted)) / len(predicted)
            if debug:
                print(f"[DEBUG] prediction stats: range=[{min_val:.3f},{max_val:.3f}] avg_diff={avg_diff:.3f} unique_ratio={unique_ratio:.3f} reasonableness={prediction_reasonableness:.3f}")
        else:
            if debug:
                print("[DEBUG] prediction stats: empty predicted")
        
        # 综合质量分数
        overall_quality = (similarity * 0.3 + 
                          predicted_quality * 0.3 + 
                          reasoning_quality * 0.2 + 
                          prediction_reasonableness * 0.2)
        
        reranked_examples.append({
            'example': example,
            'similarity': similarity,
            'predicted_quality': predicted_quality,
            'reasoning_quality': reasoning_quality,
            'prediction_reasonableness': prediction_reasonableness,
            'overall_quality': overall_quality,
            'index': example.get('index', -1)
        })
    
    # 按综合质量分数排序
    reranked_examples.sort(key=lambda x: x['overall_quality'], reverse=True)
    
    return reranked_examples


def evaluate_prediction_reasonableness(predicted_values):
    """
    评估预测结果的合理性
    """
    if len(predicted_values) == 0:
        return 0.0
    
    # 检查数值范围是否合理
    min_val = min(predicted_values)
    max_val = max(predicted_values)
    
    # 检查是否有异常值 (超出合理范围)
    reasonable_range = True
    if min_val < -50 or max_val > 50:  # 根据数据特征调整范围
        reasonable_range = False
    
    # 检查数值变化是否平滑
    smoothness_score = 0.0
    if len(predicted_values) > 1:
        diffs = [abs(predicted_values[i] - predicted_values[i-1]) for i in range(1, len(predicted_values))]
        avg_diff = sum(diffs) / len(diffs)
        # 如果平均变化幅度合理，给予高分
        if 0.1 <= avg_diff <= 5.0:
            smoothness_score = 0.8
        elif avg_diff <= 10.0:
            smoothness_score = 0.5
        else:
            smoothness_score = 0.2
    
    # 检查是否有重复值过多 (可能表示模型没有真正学习)
    unique_ratio = len(set(predicted_values)) / len(predicted_values)
    diversity_score = unique_ratio
    
    # 综合评分
    final_score = 0.0
    if reasonable_range:
        final_score += 0.4
    final_score += smoothness_score * 0.3
    final_score += diversity_score * 0.3
    
    return min(final_score, 1.0)


def construct_few_shot_prompt(base_prompt, reranked_examples, num_examples=3,use_summary=False,attr=None,data_name=None,look_back=None,pred_window=None):
    """
    构造few-shot提示，包含输入数据、推理过程和预测结果
    """
    few_shot_examples = []
    
    for i, example_info in enumerate(reranked_examples[:num_examples]):
        example = example_info['example']
        
        # 构造示例提示
        # example_prompt = f"Example {i+1} (Quality Score: {example_info['overall_quality']:.3f}):\n"
        example_prompt = f"Example {i+1}\n"
        # example_prompt += f"Input data (past {look_back} time points):\n"
        # example_prompt += example['data_string']
        # example_prompt += f"\nGround Truth (future {pred_window} time points):\n"
        # example_prompt += example['ground_truth']
        idx = example['index']
        if use_summary:
            summary = None
            summary_path = example.get("summary_file")
            if summary_path and os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)

            if isinstance(summary, dict) and summary.get('overall_summary') is not None:
                example_prompt += f"\nSummary of the reasoning process:\n{summary.get('overall_summary', '')}"
                method_dist = summary.get('method_distribution')
                if method_dist:
                    if isinstance(method_dist, dict):
                        example_prompt += "\nMethod distribution:\n" + "\n".join([f"- {k}: {v}" for k, v in method_dist.items()])
                    else:
                        example_prompt += "\nMethod distribution:\n" + str(method_dist)
                common_patterns = summary.get('common_data_patterns')
                if common_patterns:
                    if isinstance(common_patterns, list):
                        example_prompt += "\nCommon data patterns:\n" + "\n".join([f"- {item}" for item in common_patterns])
                    else:
                        example_prompt += "\nCommon data patterns:\n" + str(common_patterns)
                key_insights = summary.get('key_insights')
                if key_insights:
                    if isinstance(key_insights, list):
                        example_prompt += "\nKey insights:\n" + "\n".join([f"- {item}" for item in key_insights])
                    else:
                        example_prompt += "\nKey insights:\n" + str(key_insights)
                complexity_analysis = summary.get('complexity_analysis')
                if complexity_analysis:
                    example_prompt += "\nComplexity analysis:\n" + str(complexity_analysis)
            elif isinstance(summary, dict) and ("good_cases_distilled" in summary or "bad_cases_distilled" in summary):
                threshold = summary.get("threshold")
                good_items = summary.get("good_cases_distilled") or []
                bad_items = summary.get("bad_cases_distilled") or []
                data_pattern = None
                for candidate_items in (good_items, bad_items):
                    if not (isinstance(candidate_items, list) and candidate_items):
                        continue
                    candidate_entry = candidate_items[0] if candidate_items else None
                    if not isinstance(candidate_entry, dict):
                        continue
                    dp = candidate_entry.get("data_pattern")
                    if isinstance(dp, str) and dp.strip():
                        data_pattern = dp.strip()
                        break
                    distilled = candidate_entry.get("distilled")
                    if isinstance(distilled, list):
                        for d in distilled:
                            if not isinstance(d, dict):
                                continue
                            dp = d.get("data_pattern")
                            if isinstance(dp, str) and dp.strip():
                                data_pattern = dp.strip()
                                break
                    if data_pattern:
                        break

                # data_pattern 在 distillation 时被单独保存到 pattern 文件，summary JSON 中不含该字段
                # 若 summary JSON 中未能提取到，回退到 pattern 文件
                if not data_pattern and data_name and attr and look_back is not None and pred_window is not None:
                    _pattern_path = os.path.join(
                        "Memory", "pattern", "ETTh", data_name,
                        f"pattern_{attr}_{data_name}_{look_back}_{pred_window}_{idx}.json"
                    )
                    if os.path.exists(_pattern_path):
                        try:
                            with open(_pattern_path, 'r', encoding='utf-8') as _fp:
                                _pat = json.load(_fp)
                            _dp = _pat.get("data_pattern", "")
                            if isinstance(_dp, str) and _dp.strip():
                                data_pattern = _dp.strip()
                        except Exception:
                            pass

                if data_pattern:
                    example_prompt += f"\nData pattern:\n{data_pattern}"

                example_prompt += "\nDistilled lessons (good/bad case based):"
                if threshold is not None:
                    example_prompt += f"\nThreshold (MSE q={summary.get('quantile', 'N/A')}): {threshold}"
                items = [("good", good_items), ("bad", bad_items)]

                for case_type, case_items in items:
                    if not case_items:
                        continue
                    case_entry = case_items[0] if isinstance(case_items, list) and case_items else None
                    if not isinstance(case_entry, dict):
                        continue
                    distilled = case_entry.get("distilled")
                    if not isinstance(distilled, list):
                        continue
                    for d in distilled[:3]:
                        if not isinstance(d, dict):
                            continue
                        if case_type == "good":
                            whentouse = d.get("when_to_use")
                            experience = d.get("experience")
                            success_insight = d.get("success_insight")
                            meta_logic = d.get("meta_logic")
                            meta_logic_str = json.dumps(meta_logic, ensure_ascii=False) if isinstance(meta_logic, dict) else str(meta_logic or "")
                            text = (
                                f"when_to_use: {whentouse}\n"
                                f"experience: {experience}\n"
                                f"success_insight: {success_insight}\n"
                                f"meta_logic: {meta_logic_str}"
                            ).strip()
                        else:
                            whentouse = d.get("when_to_use")
                            failure_analysis = d.get("failure_analysis")
                            preventative_rule = d.get("preventative_rule")
                            meta_logic = d.get("meta_logic")
                            meta_logic_str = json.dumps(meta_logic, ensure_ascii=False) if isinstance(meta_logic, dict) else str(meta_logic or "")
                            text = (
                                f"when_to_use: {whentouse}\n"
                                f"failure_analysis: {failure_analysis}\n"
                                f"preventative_rule: {preventative_rule}\n"
                                f"meta_logic: {meta_logic_str}"
                            ).strip()
                        if not text.strip() or text.strip() == "meta_logic:":
                            continue
                        tags = d.get("tags")
                        if isinstance(tags, list) and tags:
                            example_prompt += f"\n- ({case_type}) [{', '.join([str(t) for t in tags])}] {text}"
                        else:
                            example_prompt += f"\n- ({case_type}) {text}"
            else:
                example_prompt += "\nSummary not found or invalid."
        else:
            # 从 pattern 文件加载 data_pattern
            if data_name and attr and look_back is not None and pred_window is not None:
                _pattern_path = os.path.join(
                    "Memory", "pattern", "ETTh", data_name,
                    f"pattern_{attr}_{data_name}_{look_back}_{pred_window}_{idx}.json"
                )
                if os.path.exists(_pattern_path):
                    try:
                        with open(_pattern_path, 'r', encoding='utf-8') as _fp:
                            _pat = json.load(_fp)
                        _dp = _pat.get("data_pattern", "")
                        if isinstance(_dp, str) and _dp.strip():
                            example_prompt += f"\nData pattern:\n{_dp.strip()}"
                    except Exception:
                        pass
            # 如果有推理过程，添加推理过程
            if example.get('reasoning_string', '').strip() and example.get('reasoning_string', '') != 'Error':
                example_prompt += f"\nReasoning process:\n{example['reasoning_string'][:500]}..."  # 限制长度
                if len(example.get('reasoning_string', '')) > 500:
                    example_prompt += "\n[Reasoning continues...]"
        
        # 添加预测结果
        if example.get('forecast_string', '').strip():
            example_prompt += f"\nForecast result:\n{example['forecast_string']}"
        
        example_prompt += "\n" + "="*60 + "\n"
        
        few_shot_examples.append(example_prompt)
    
    # 构造完整的few-shot提示
    few_shot_prompt = "Here are some similar examples from the training set to help you understand the task:\n\n"
    few_shot_prompt += "\n".join(few_shot_examples)
    # few_shot_prompt += (  
    #     "\n\nBefore forecasting, perform a brief meta-analysis using the similar examples above:\n"
    #     "1) For each example, analyze the future ground-truth trajectory and decide whether a change point occurs (Yes/No).\n"
    #     "2) Classify the example's current regime as one of: stable / trending / post-shock recovery.\n"
    #     "3) For the next pred_len (forecast horizon), judge which outcome is more likely: mean reversion / continued drawdown / slow recovery.\n"
    #     "4) Then analyze the current window and infer the most likely regime and trend direction for the next pred_len.\n"
    #     "Use the above decisions to guide the shape of your forecast (turning points, slopes, recovery speed), without copying exact values.\n"
    # )
    
    # few_shot_prompt += "\nSimilar examples show the trend characterized by an abrupt, rapid decline from a peak or stabilization period, followed by a gradual, oscillating long-tail recovery toward a new equilibrium.Please follow the pattern while allowing for differences in magnitude.\n"
    few_shot_prompt += "\nBased on these examples, please solve the following similar task:\n\n"
    few_shot_prompt += base_prompt
    
    return few_shot_prompt


def ETTh_main_few_shot_reasoning(data_name, attr, look_back, pred_window, number, api_key, 
                                 temperature=0.6, top_p=0.7, num_similar_examples=10, 
                                 num_few_shot_examples=3,use_summary=False,
                                 _vision_analysis=True,
                                 _covariate_data=True):
    """
    主函数：实现相似样例检索 + 质量再排序 + few-shot提示构造的one-shot推理
    从Memory中存储的训练集结果中检索相似案例
    """

    result_dir_uq = f'results/ETTh/result_uq/{data_name}'
    result_dir_best = f'results/ETTh/result_best/{data_name}'
    
    # 加载数据
    data_dir = './datasets/' + data_name + '.csv'
    data = pd.read_csv(data_dir)

    # 取前16个月的数据
    # data = data[12 * 30 * 24 + 4 * 30 * 24 - look_back : 12 * 30 * 24 + 8 * 30 * 24]
    # date = data.loc[:, 'date'].to_numpy()
    # attr_data = data.loc[:, attr].to_numpy()
    # data = pd.DataFrame(date, columns=['date'])
    # data[attr] = attr_data
    end_idx = int(len(data) * 0.8)
    end = look_back + pred_window
    stride = 24
    max_start = end_idx - end + 1
    starts = list(range(0, max_start, stride))
    max_trainvalid_samples = len(starts)
    saved_number = max_trainvalid_samples + number

    result_file_best = f'{result_dir_best}/result_{attr}_{data_name}_{look_back}_{pred_window}_{saved_number}.json'
    if os.path.exists(result_file_best):
        print(f"Existing result found at {result_file_best}. Skipping.")
        return
    
    # 取测试集的部分
    start_idx = int(len(data) * 0.8) - look_back
    if start_idx < 0:
        start_idx = 0
    data_test = data[start_idx:]
    
    # 保存原始attr用于prompt显示
    original_attr = attr
    
    # 获取所有三个变量的列名映射（原始名称 -> 实际列名）
    all_attrs_dict = {
        'HUFL': 'HUFL',
        'HULL': 'HULL',
        'MUFL': 'MUFL',
        'MULL': 'MULL',
        'LUFL': 'LUFL',
        'LULL': 'LULL',
        'OT': 'OT',
        'tmax': 'tmax',
        'tmin': 'tmin'
    }
    
    # 检查并修正列名映射
    for key, value in all_attrs_dict.items():
        if value not in data.columns:
            # 尝试查找匹配的列
            matching_cols = [col for col in data.columns if value.lower().strip() in col.lower().strip() or col.lower().strip() in value.lower().strip()]
            if matching_cols:
                all_attrs_dict[key] = matching_cols[0]
    
    # 获取目标变量的实际列名
    if attr not in data.columns:
        # 尝试查找匹配的列
        matching_cols = [col for col in data.columns if attr.lower().strip() in col.lower().strip() or col.lower().strip() in attr.lower().strip()]
        if matching_cols:
            attr_col = matching_cols[0]
        else:
            # 尝试从all_attrs_dict中查找
            attr_col = all_attrs_dict.get(attr, attr)
    else:
        attr_col = attr
        
    image_path = f'./Memory/visual/image/ETTh/{data_name}/test/{data_name}_visualization_{number:03d}.png'
    meta_path = image_path.replace('.png', '.json')
    selected_covs = []
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            covs = meta.get('covariates', [])
            if isinstance(covs, list) and len(covs) > 0:
                selected_covs = covs[:2]
        except:
            selected_covs = []
    if not selected_covs:
        fallback = [k for k in ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'] if k in data.columns and k != original_attr]
        selected_covs = fallback[:2] if fallback else []
        
    # 强制添加温度协变量
    for temp_cov in ['tmax', 'tmin']:
        if temp_cov in data.columns and temp_cov not in selected_covs:
            selected_covs.append(temp_cov)
            
    covariates = {k: all_attrs_dict.get(k, k) for k in selected_covs if all_attrs_dict.get(k, k) in data.columns}
        
    # 构建包含目标变量和协变量的完整数据框
    data_full = pd.DataFrame(data['date'].values, columns=['date'])
    if attr_col in data.columns:
        data_full[original_attr] = data[attr_col].values
        
    data_test_full = pd.DataFrame(data_test['date'].values, columns=['date'])
    if attr_col in data_test.columns:
        data_test_full[original_attr] = data_test[attr_col].values
        
    # 添加协变量（使用原始名称作为列名，便于理解）
    for cov_key, cov_col in covariates.items():
        data_full[cov_key] = data[cov_col].values
        data_test_full[cov_key] = data_test[cov_col].values
        
    # 调试信息：打印实际包含的列
    print(f"Target variable: {original_attr} (column: {attr_col})")
    print(f"Covariates: {list(covariates.keys())}")
    print(f"Data columns in output: {list(data_full.columns)}")
    
    # 滑动窗口设置：每次滑动24个时间步
    slide_window = 24
    data_lookback_test = []
    data_lookback = []
    max_samples = (len(data_full) - look_back) // slide_window + 1
    max_test_samples = (len(data_test_full) - look_back) // slide_window + 1
    for i in range(min(30, max_test_samples)):
        start_idx = i * slide_window
        end_idx = start_idx + look_back
        if end_idx <= len(data_test_full):
            data_lookback_test.append(data_test_full.iloc[start_idx:end_idx])
            
    for i in range(max_samples):
        start_idx = i * slide_window
        end_idx = start_idx + look_back
        if end_idx <= len(data_full):
            data_lookback.append(data_full.iloc[start_idx:end_idx])

    # 计算当前窗口的历史数据结束位置
    current_start_idx = number * slide_window
    current_end_idx = current_start_idx + look_back
    
    # 分别构建目标变量和协变量的数据框
    target_data = pd.DataFrame({
        'date': data_lookback_test[number]['date'],
        original_attr: data_lookback_test[number][original_attr]
    })
    
    # 准备历史样例数据 - 从Memory中加载训练集结果
    # data_lookback = []
    historical_examples = []
    
    # 构建协变量数据框（如果有协变量）
    covariate_data = None
    if covariates:
        cov_cols = ['date'] + list(covariates.keys())
        covariate_data = data_lookback_test[number][cov_cols].copy()
        
    # 提取未来的协变量数据
    future_covariate_data = None
    if covariates and current_end_idx + pred_window <= len(data_test_full):
        future_start_idx = current_end_idx
        future_end_idx = current_end_idx + pred_window
        future_cov_cols = ['date'] + list(covariates.keys())
        future_covariate_data = data_test_full.iloc[future_start_idx:future_end_idx][future_cov_cols].copy()
        
    vision_analysis = None
    if _vision_analysis:
        # 尝试加载视觉分析的推理结果（如果存在）
        vision_analysis = None
        vision_result_path_etth = f'Memory/visual/analysis/ETTh/{data_name}/test/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json'
        chosen_vision_path = vision_result_path_etth
        if os.path.exists(chosen_vision_path):
            try:
                with open(chosen_vision_path, 'r') as f:
                    vision_results = json.load(f)
                    if vision_results and len(vision_results) > 0:
                        vision_answer_str = vision_results[0].get('answer', '')
                        if vision_answer_str:
                            # answer字段本身是一个JSON字符串，需要再次解析
                            vision_analysis = json.loads(vision_answer_str)
                            print(f"Loaded vision analysis: {list(vision_analysis.keys())}")
            except Exception as e:
                print(f"Warning: Could not load vision analysis from {chosen_vision_path}: {e}")
                vision_analysis = None
        else:
            print(f"Vision analysis file not found: {chosen_vision_path}")

    entry_map = {}
    meta_path = os.path.join("Memory", "cases", "ETTh", "vector_db", data_name, f"db_{attr}_{look_back}_{pred_window}.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            entries = meta.get("entries", []) if isinstance(meta, dict) else []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                summary_file = entry.get("summary_file")
                if not summary_file or not os.path.exists(summary_file):
                    continue
                sample_id = entry.get("id", "")
                try:
                    entry_index = int(str(sample_id).split(":")[-1])
                except Exception:
                    continue
                entry_map[entry_index] = entry
        except Exception:
            entry_map = {}

    max_hist = min(saved_number, len(data_lookback))
    for example_index in range(max_hist):
        try:
            if example_index not in entry_map:
                continue
            entry = entry_map[example_index]
            example_data = data_lookback[example_index]
            target_col_for_example = original_attr if original_attr in example_data.columns else attr

            if len(example_data) == 0 or example_data[target_col_for_example].isna().all():
                continue

            gt_start = example_index * slide_window + look_back
            gt_end = gt_start + pred_window
            if gt_end > len(data_full):
                continue

            gt_df = data_full.iloc[gt_start:gt_end][['date', original_attr]]
            if len(gt_df) != pred_window or gt_df[original_attr].isna().any():
                continue

            # 构建样例输入数据字符串（包含协变量）
            cols_to_include = ['date', target_col_for_example]
            for cov_key in covariates.keys():
                if cov_key in example_data.columns:
                    cols_to_include.append(cov_key)
            example_data_for_prompt = example_data[cols_to_include]
            
            # 构建样例真实值数据字符串（包含协变量）
            gt_cols_to_include = ['date', original_attr]
            for cov_key in covariates.keys():
                if cov_key in data_full.columns:
                    gt_cols_to_include.append(cov_key)
            gt_df_for_prompt = data_full.iloc[gt_start:gt_end][gt_cols_to_include]

            historical_examples.append({
                'data': example_data[target_col_for_example].values,
                'data_string': example_data_for_prompt.to_string(index=False),
                'ground_truth': gt_df_for_prompt.to_string(index=False),
                'index': example_index,
                'summary_file': entry.get("summary_file"),
                'confidence': entry.get("confidence", 0)
            })
        except Exception as e:
            print(f"Error processing data for index {example_index}: {e}")
            continue

    print(f"Loaded {len(historical_examples)} historical examples (index < {saved_number})")
    try:
        sample_count = min(30, len(historical_examples))
        if debug:
            print(f"[DEBUG] sample_count: {sample_count}")
        for i in range(sample_count):
            ex = historical_examples[i]
            vals = ex['data']
            feats = extract_time_series_features(vals)
            head_vals = vals[:10]
            if debug:
                print(f"[DEBUG] hist idx={ex['index']} len={len(vals)} head={head_vals}")
                print(f"[DEBUG] features: {feats}")
    except Exception as e:
        if debug:
            print(f"[DEBUG] error printing historical samples: {e}")
    
    # 获取当前要预测的数据
    try:
        # current_data = data.iloc[number * look_back:(number + 1) * look_back]
        # current_values = current_data[attr].values
        
        current_data = target_data
        current_values = current_data[attr].values
        
        
        # 检查当前数据是否有效
        if len(current_data) == 0 or current_data[attr].isna().all():
            print(f"Error: Invalid current data for index {number}")
            return
            
        print(f"Current data shape: {current_data.shape}, values range: [{current_values.min():.3f}, {current_values.max():.3f}]")
    except Exception as e:
        print(f"Error loading current data: {e}")
        return
    
    # 1. 相似样例检索
    print("Step 1: Retrieving similar examples...")
    
    # Method 1: Cosine
    print("Method 1: Cosine Similarity")
    similar_examples_cosine = retrieve_similar_examples(current_values, historical_examples, 
                                                top_k=num_similar_examples)
    reranked_examples_cosine = similar_examples_cosine[:num_few_shot_examples]
    print(f"Found {len(similar_examples_cosine)} similar examples (Cosine)")
    
    # Method 2: DTW
    print("Method 2: DTW")
    similar_examples_dtw = retrieve_similar_examples_dtw(current_values, historical_examples, 
                                                top_k=num_similar_examples)
    reranked_examples_dtw = similar_examples_dtw[:num_few_shot_examples]
    print(f"Found {len(similar_examples_dtw)} similar examples (DTW)")

    # 打印部分信息
    print("Cosine Examples Indices:", [ex['index'] for ex in reranked_examples_cosine])
    print("DTW Examples Indices:", [ex['index'] for ex in reranked_examples_dtw])
        
    
    # 3. 构造基础提示
    # base_prompt = f'You are an expert in time series forecasting.\n'
    # base_prompt += f'The dataset records oil temperature and load metrics from electricity transformers, tracked between July 2016 and July 2018. '
    # base_prompt += f'It is subdivided into four mini-datasets, with data sampled either hourly or every 15 minutes. '
    # base_prompt += f'Here is the {meaning_dict[attr]} data of the transformer.\n'
    # base_prompt += f'I will now give you data for the past {look_back} recorded dates, and please help me forecast the data for next {pred_window} recorded dates.\n'
    # base_prompt += f'But please note that these data will have missing values, so be aware of that.\n'
    # base_prompt += f'Please give me the complete data for the next {pred_window} recorded dates, remember to give me the complete data.\n'
    # base_prompt += f'You must provide the complete data. You mustn\'t omit any content.\n'
    # base_prompt += f'The data is as follows:\n'
    # base_prompt += current_data.to_string(index=False)
    # base_prompt += f'\nAnd your final answer must follow the format:\n'
    # base_prompt += """
    # <answer>
    #     \\n```\\n
    #     ...
    #     \\n```\\n
    #     </answer>
    # Please obey the format strictly. And you must give me the complete answer.
    # """
    base_prompt = ''
    base_prompt = f'You are an expert in time series forecasting.\n'
    base_prompt += ' The dataset is ETTh1 (Electricity Transformer Temperature Hourly) and contains hourly records of transformer oil temperature (target) and load-related covariates (HUFL, HULL, MUFL, MULL, LUFL, LULL).'
    base_prompt += f'\n\n**Target Variable to Forecast:**\n'
    base_prompt += f'Here is the historical data of {meaning_dict.get(original_attr, original_attr)} (the variable you need to forecast) for the past {look_back} recorded hours:\n'
    # base_prompt += f'Here is the historical data of (the variable you need to forecast) for the past {look_back} recorded hours:\n'
    base_prompt += target_data.to_string(index=False)
    
    # 单独展示协变量
    # print(_covariate_data, covariates, covariate_data)
    if _covariate_data and covariate_data is not None and len(covariates) > 0:
        base_prompt += f'\n\n**Covariates (Auxiliary Variables):**\n'
        cov_meanings_list = []
        for cov_key in covariates.keys():
            cov_meanings_list.append(f"{cov_key}: {meaning_dict.get(cov_key, cov_key)}")
        base_prompt += 'I also provide the following covariates (auxiliary variables) that may help with forecasting. '
        base_prompt += f'These covariates represent different frequency components of electrical load and may relate to oil temperature dynamics:\n'
        base_prompt += '\n'.join(cov_meanings_list)
        base_prompt += f'\n\nHistorical data of covariates for the past {look_back} recorded hours:\n'
        base_prompt += covariate_data.to_string(index=False)
        
        # 添加未来的协变量数据
        if future_covariate_data is not None:
            base_prompt += f'\n\n**Future Covariates (Known Future Values):**\n'
            base_prompt += f'I also provide the future values of the covariates for the next {pred_window} recorded hours. '
            base_prompt += f'These are known forecasted values that you can use to help predict the target variable:\n'
            base_prompt += future_covariate_data.to_string(index=False)

    base_prompt += "\n\n**Statistical Summary (Feature Extraction):**\n"
    base_prompt += _format_feature_stats(f"Target {original_attr}", target_data[original_attr].values)
    if _covariate_data and covariate_data is not None and len(covariates) > 0:
        for cov_key in covariates.keys():
            if cov_key in covariate_data.columns:
                base_prompt += "\n" + _format_feature_stats(f"Covariate {cov_key} (past)", covariate_data[cov_key].values)
        if future_covariate_data is not None:
            for cov_key in covariates.keys():
                if cov_key in future_covariate_data.columns:
                    base_prompt += "\n" + _format_feature_stats(f"Covariate {cov_key} (future)", future_covariate_data[cov_key].values)
    
    # 添加视觉分析的推理结果（如果存在）
    if _vision_analysis and vision_analysis is not None:
        base_prompt += f'\n\n**Time-Series Analysis (Visual Reasoning Insights):**\n'
        base_prompt += 'Based on visual analysis of the time-series patterns, here are key insights that may help with forecasting:\n\n'
        
        if 'trend' in vision_analysis:
            base_prompt += f'- **Trend**: {vision_analysis["trend"]}\n'
        if 'seasonality' in vision_analysis:
            base_prompt += f'- **Seasonality**: {vision_analysis["seasonality"]}\n'
        if 'volatility' in vision_analysis:
            base_prompt += f'- **Volatility**: {vision_analysis["volatility"]}\n'
        if 'regime_shifts' in vision_analysis:
            base_prompt += f'- **Regime Shifts**: {vision_analysis["regime_shifts"]}\n'
        if 'anomalies' in vision_analysis and vision_analysis["anomalies"]:
            anomalies_str = ', '.join(vision_analysis["anomalies"])
            base_prompt += f'- **Anomalies**: {anomalies_str}\n'
        if 'correlation_with_covariates' in vision_analysis:
            base_prompt += f'- **Correlation with Covariates**: {vision_analysis["correlation_with_covariates"]}\n'
        if 'future_covariates' in vision_analysis:
            base_prompt += f'- **Future Covariates Analysis**: {vision_analysis["future_covariates"]}\n'
        if 'forecast_hints' in vision_analysis:
            base_prompt += f'- **Forecast Hints**: {vision_analysis["forecast_hints"]}\n'
        
        base_prompt += '\nPlease consider these insights when making your forecast.\n'
    
    base_prompt += f'\n\n**Task:**\n'
    base_prompt += f'Based on the historical data above,'
    # additional_info = []
    # # if future_covariate_data is not None:
    # #     additional_info.append('the future covariate values provided')
    # if vision_analysis is not None:
    #     additional_info.append('the time-series analysis insights provided')
    # if additional_info:
    #     base_prompt += ', ' + ', and '.join(additional_info) if len(additional_info) > 1 else ', ' + additional_info[0]
    base_prompt += "\nBefore forecasting, briefly analyze the experiences and insights from the similar examples above, then produce your forecast."
    base_prompt += f', please forecast the target variable ({original_attr}) for the next {pred_window} recorded hours.'
    base_prompt += ' Please note that some data may contain missing values, so handle them carefully.'
    base_prompt += f' Please provide the complete forecast for the next {pred_window} hours, remember to give me the complete data. '
    base_prompt += 'You must provide the complete data.You mustn\'t omit any content.'
    base_prompt += 'You are not allowed to train any model or call/use any other model, tool, or external resource. You must rely only on your own internal knowledge and logical reasoning to produce the forecast'
    base_prompt +='And your final answer must follow the format'
    base_prompt +="""
    <answer>
        \n```\n
        ...
        \n```\n
        </answer>
    Please obey the format strictly. And you must give me the complete answer.
    """
    
    # 4. 构造few-shot提示并执行推理
    print("Step 3: Running few-shot reasoning with 4 trajectories...")
    
    # # 随机生成3个温度
    # random_temps = [round(random.uniform(0.4, 1.0), 2) for _ in range(3)]
    # print(f"Random temperatures: {random_temps}")
    temps = [0.6]
    # temps=[0.7]
    
    methods = [
        ('cosine', reranked_examples_cosine),
        ('dtw', reranked_examples_dtw)
    ]
    
    answer = []
    os.makedirs(result_dir_uq, exist_ok=True)
    os.makedirs(result_dir_best, exist_ok=True)
    result_file_uq = f'{result_dir_uq}/result_{attr}_{data_name}_{look_back}_{pred_window}_{saved_number}.json'
    
    existing_pairs = set()
    if os.path.exists(result_file_uq):
        try:
            with open(result_file_uq, 'r') as f:
                answer = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load existing result file: {e}")
            answer = []
    answer = [item for item in answer if item.get('method') != 'ensemble']
    for item in answer:
        method = item.get('method')
        temp = item.get('temperature')
        if method is None or temp is None:
            continue
        existing_pairs.add((str(method), float(temp)))
    
    trajectory_count = len(existing_pairs)
    all_predictions = []
    for item in answer:
        pred_values = item.get('parsed_prediction')
        if isinstance(pred_values, list) and len(pred_values) > 0:
            all_predictions.append(pred_values)
    
    for method_name, examples in methods:
        # Construct prompt for this method
        print(f"Constructing prompt for method: {method_name}")
        few_shot_prompt = construct_few_shot_prompt(base_prompt, examples, 
                                                num_examples=num_few_shot_examples,use_summary=use_summary,attr=attr,data_name=data_name,look_back=look_back,pred_window=pred_window)
        
        # Save prompt
        prompt_dir = f'Memory/predictions/ETTh/prompt/{data_name}'
        os.makedirs(prompt_dir, exist_ok=True)
        with open(f'{prompt_dir}/prompt_{attr}_{data_name}_{look_back}_{pred_window}_{saved_number}_{method_name}.txt', 'w') as f: 
            f.write(few_shot_prompt)
        
        for temp in temps:
            if (str(method_name), float(temp)) in existing_pairs:
                print(f"Skip existing method={method_name}, temp={temp}")
                continue
            trajectory_count += 1
            print(f"Trajectory {trajectory_count}/4: Method={method_name}, Temp={temp}")
            
            model = deepseek_api_output(api_key=api_key, temperature=temp, top_p=top_p)
            reasoning, result = _run_with_reflection(
                model=model,
                prompt=few_shot_prompt,
                pred_window=pred_window,
                historical_values=current_values,
            )
            
            # Parse result
            pred_values = get_result(result)
            if len(pred_values) > 0:
                all_predictions.append(pred_values)
            
            answer.append({
                'index': trajectory_count - 1,
                'method': method_name,
                'temperature': temp,
                'reasoning': reasoning,
                'answer': result,
                'parsed_prediction': pred_values,
                'similar_examples_used': [
                    {
                        'index': ex['index'],
                        'similarity': ex['similarity']
                    } for ex in examples
                ]
            })
            
    with open(result_file_uq, 'w') as f:
        json.dump(answer, f, indent=4)
    print(f"Saved trajectories to {result_file_uq}")

    trajectory_items = [item for item in answer if item.get('method') != 'ensemble']
    trajectory_values = []
    for item in trajectory_items:
        values = item.get('parsed_prediction')
        if isinstance(values, list) and len(values) > 0:
            trajectory_values.append(values)
        else:
            trajectory_values.append([])
    best_idx, similar_meta, llm_answer = select_best_trajectory_with_llm(
        current_values,
        historical_examples,
        trajectory_values,
        api_key,
        data_name,
        attr,
        look_back,
        pred_window
    )
    if best_idx is None:
        best_idx = 0 if trajectory_items else None
        selection_method = "fallback"
    else:
        selection_method = "llm"
    best_entry = None
    if best_idx is not None and 0 <= best_idx < len(trajectory_items):
        best_entry = dict(trajectory_items[best_idx])
        best_entry['selection_method'] = selection_method
        best_entry['llm_answer'] = llm_answer
        best_entry['llm_similar_examples'] = similar_meta
    else:
        best_entry = {
            'index': None,
            'method': None,
            'temperature': None,
            'reasoning': None,
            'answer': None,
            'parsed_prediction': [],
            'selection_method': selection_method,
            'llm_answer': llm_answer,
            'llm_similar_examples': similar_meta
        }

    with open(result_file_best, 'w') as f:
        json.dump([best_entry], f, indent=4)

    # Analyze and add to dynamic memory
    print("Step 4: Analyzing memory reasoning and updating dynamic memory...")
    infer_similar_ids = []
    for ex in reranked_examples_cosine[:num_few_shot_examples]:
        idx = ex.get("index")
        if isinstance(idx, int):
            infer_similar_ids.append(f"{data_name}:{attr}:{look_back}:{pred_window}:{idx}")
    try:
        analyze_memory_reasoning(
            data_name=data_name,
            api_key=api_key,
            attr=attr,
            look_back=look_back,
            pred_window=pred_window,
            number=saved_number,
            output_dir='Memory/cases/summary/ETTh',
            overwrite=True,
            dedup_enable=True,
            train_ratio=1.0,
            debug_trace=True,
            infer_mode=True,
            infer_similar_ids=infer_similar_ids
        )
        prompt_dir = os.path.join("Memory", "cases", "origin", "ETTh", "prompt_cat")
        prompt_good = os.path.join(prompt_dir, f"prompt_good_{data_name}_{attr}_{look_back}_{pred_window}_{saved_number}.txt")
        prompt_bad = os.path.join(prompt_dir, f"prompt_bad_{data_name}_{attr}_{look_back}_{pred_window}_{saved_number}.txt")
        print(f"[TRACE] prompt_good_exists={os.path.exists(prompt_good)} path={prompt_good}")
        print(f"[TRACE] prompt_bad_exists={os.path.exists(prompt_bad)} path={prompt_bad}")
        db_path = os.path.join("Memory", "cases", "ETTh", "vector_db", data_name, f"db_{attr}_{look_back}_{pred_window}.npz")
        meta_path = os.path.splitext(db_path)[0] + ".json"
        print(f"[TRACE] vector_db_exists={os.path.exists(db_path) and os.path.exists(meta_path)} db={db_path}")
        print("Memory analysis completed.")
    except Exception as e:
        print(f"Error during memory analysis: {e}")
    
    print('Few-shot reasoning completed!')


if __name__ == "__main__":
    # 示例调用
    ETTh_main_few_shot_reasoning(
        data_name='ETTh1',
        attr='HUFL',
        look_back=96,
        pred_window=96,
        number=0,
        api_key='your-api-key-here',
        temperature=0.6,
        top_p=0.7,
        num_similar_examples=10,
        num_few_shot_examples=3,
        use_summary=True
    )
