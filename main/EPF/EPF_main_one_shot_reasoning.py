import pandas as pd
# from utils.api_ouput import qianduoduo_api_output
# from utils.api_ouput import nvidia_api_output
from utils.api_ouput import deepseek_api_output
from utils.api_ouput import gpt4_api_output
import json
import os
import re
import time
import numpy as np

meaning_dict = {
    'Grid load forecast': 'Forecasted grid electricity load in the Nord Pool system (MW)',
    'Wind power forecast': 'Forecasted wind power generation in the Nord Pool system (MW)',
    'OT': 'Observed Nord Pool electricity price (€/MWh, 1-hour resolution)'
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
    return os.path.join("Memory", "cases", "EPF", "summary_outputs", data_name)


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
    model = deepseek_api_output(api_key=api_key, temperature=0.2, top_p=0.7)
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

def _extract_forecast_lines(answer_text):
    """
    Extract forecast lines from the model answer.

    Returns:
        tuple[list[str] | None, str]: (lines, error_message)
    """
    if not answer_text or not answer_text.strip():
        return None, "Answer text is empty."

    match = re.search(r"<answer>\s*```(.*?)```\s*</answer>", answer_text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None, "Answer does not follow the required <answer> ```...``` </answer> format."

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

    if len(lines) != pred_window:
        issues.append(f"Expected {pred_window} forecast lines but found {len(lines)}.")

    timestamps = set()

    for idx, line in enumerate(lines, start=1):
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


def EPF_main_one_shot_reasoning_see(
    data_name,
    attr,
    look_back,
    pred_window,
    number,
    api_key,
    temperature,
    top_p,
    max_retries=3,
    max_duration_seconds=180,
    retry_delay_seconds=3,
    volatility_ratio_threshold=1.3
):
    _covariate_data=True

    data_dir = './datasets/' + data_name + '.csv'
    data = pd.read_csv(data_dir)
    # 清理列名，去除可能的空格
    data.columns = data.columns.str.strip()
    data['date'] = pd.to_datetime(data['date'])
    
    # 取前12个月的数据 (假设每小时一个数据点，12个月 = 12 * 30 * 24)
    end_idx = int(len(data) * 0.8)
    if end_idx > len(data):
        end_idx = len(data)
    data = data[:end_idx]
    
    # # 取测试集的部分
    # start_idx = int(len(data) * 0.8) - look_back
    # if start_idx < 0:
    #     start_idx = 0
    # data = data[start_idx:]
    
    # 取训练集 + 验证集的部分（按时间顺序取 70%~80% 区间）
    # start_idx = int(len(data) * 0.7) - look_back
    # if start_idx < 0:
    #     start_idx = 0
    # end_idx = int(len(data) * 0.8)
    # if end_idx > len(data):
    #     end_idx = len(data)
    # data = data[:end_idx]
    
    # 保存原始attr用于prompt显示
    original_attr = attr
    
    # 获取所有三个变量的列名映射（原始名称 -> 实际列名）
    all_attrs_dict = {
        'Grid load forecast': 'Grid load forecast',
        'Wind power forecast': 'Wind power forecast',
        'OT': 'OT'
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
    
    image_path = f'./Memory/visual/image/EPF/{data_name}/memory/{data_name}_visualization_{number:03d}.png'
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
        fallback = [k for k in ['Grid load forecast', 'Wind power forecast', 'OT'] if k in data.columns and k != original_attr]
        selected_covs = fallback[:2] if fallback else []
    covariates = {k: all_attrs_dict.get(k, k) for k in selected_covs if all_attrs_dict.get(k, k) in data.columns}
    
    # 构建包含目标变量和协变量的完整数据框
    data_full = pd.DataFrame(data['date'].values, columns=['date'])
    if attr_col in data.columns:
        data_full[original_attr] = data[attr_col].values
    
    # 添加协变量（使用原始名称作为列名，便于理解）
    for cov_key, cov_col in covariates.items():
        data_full[cov_key] = data[cov_col].values
    
    # 调试信息：打印实际包含的列
    print(f"Target variable: {original_attr} (column: {attr_col})")
    print(f"Covariates: {list(covariates.keys())}")
    print(f"Data columns in output: {list(data_full.columns)}")
    
    # 滑动窗口设置：每次滑动24个时间步
    slide_window = 24
    data_lookback = []
    max_samples = (len(data_full) - look_back) // slide_window + 1
    # for i in range(min(30, max_samples)):
    for i in range(max_samples):
        start_idx = i * slide_window
        end_idx = start_idx + look_back
        if end_idx <= len(data_full):
            data_lookback.append(data_full.iloc[start_idx:end_idx])

    # number 为顺序样本索引（0-29），映射到训练集最后 30 个窗口
    actual_idx = max(len(data_lookback) - 30, 0) + number

    # 计算当前窗口的历史数据结束位置
    current_start_idx = actual_idx * slide_window
    current_end_idx = current_start_idx + look_back

    # 分别构建目标变量和协变量的数据框
    target_data = pd.DataFrame({
        'date': data_lookback[actual_idx]['date'],
        original_attr: data_lookback[actual_idx][original_attr]
    })

    # 构建协变量数据框（如果有协变量）
    covariate_data = None
    if covariates:
        cov_cols = ['date'] + list(covariates.keys())
        covariate_data = data_lookback[actual_idx][cov_cols].copy()
    
    # 提取未来的协变量数据
    future_covariate_data = None
    if covariates and current_end_idx + pred_window <= len(data_full):
        future_start_idx = current_end_idx
        future_end_idx = current_end_idx + pred_window
        future_cov_cols = ['date'] + list(covariates.keys())
        future_covariate_data = data_full.iloc[future_start_idx:future_end_idx][future_cov_cols].copy()

    # 尝试加载视觉分析的推理结果（如果存在）
    vision_analysis = None
    vision_result_path_etth = f'Memory/visual/analysis/EPF/{data_name}/memory/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json'
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

    prompt = ''
    prompt += ' The dataset comes from the Nord Pool electricity market and contains hourly records of grid load forecasts, wind power forecasts, and observed electricity prices.'
    prompt += f'\n\n**Target Variable to Forecast:**\n'
    prompt += f'Here is the historical data of {meaning_dict.get(original_attr, original_attr)} (the variable you need to forecast) for the past {look_back} recorded hours:\n'
    prompt += target_data.to_string(index=False)
    
    # 单独展示协变量
    if covariate_data is not None and len(covariates) > 0:
        prompt += f'\n\n**Covariates (Auxiliary Variables):**\n'
        cov_meanings_list = []
        for cov_key in covariates.keys():
            cov_meanings_list.append(f"{cov_key}: {meaning_dict.get(cov_key, cov_key)}")
        prompt += 'I also provide the following covariates (auxiliary variables) that may help with forecasting. '
        prompt += f'These covariates capture market supply-demand drivers and may relate to price dynamics:\n'
        prompt += '\n'.join(cov_meanings_list)
        prompt += f'\n\nHistorical data of covariates for the past {look_back} recorded hours:\n'
        prompt += covariate_data.to_string(index=False)
        
        # 添加未来的协变量数据
        if future_covariate_data is not None:
            prompt += f'\n\n**Future Covariates (Known Future Values):**\n'
            prompt += f'I also provide the future values of the covariates for the next {pred_window} recorded hours. '
            prompt += f'These are known forecasted values that you can use to help predict the target variable:\n'
            prompt += future_covariate_data.to_string(index=False)
            
    prompt += "\n\n**Statistical Summary (Feature Extraction):**\n"
    prompt += _format_feature_stats(f"Target {original_attr}", target_data[original_attr].values)
    if _covariate_data and covariate_data is not None and len(covariates) > 0:
        for cov_key in covariates.keys():
            if cov_key in covariate_data.columns:
                prompt += "\n" + _format_feature_stats(f"Covariate {cov_key} (past)", covariate_data[cov_key].values)
        if future_covariate_data is not None:
            for cov_key in covariates.keys():
                if cov_key in future_covariate_data.columns:
                    prompt += "\n" + _format_feature_stats(f"Covariate {cov_key} (future)", future_covariate_data[cov_key].values)
    
    # 添加视觉分析的推理结果（如果存在）
    if vision_analysis is not None:
        prompt += f'\n\n**Time-Series Analysis (Visual Reasoning Insights):**\n'
        prompt += 'Based on visual analysis of the time-series patterns, here are key insights that may help with forecasting:\n\n'
        
        if 'trend' in vision_analysis:
            prompt += f'- **Trend**: {vision_analysis["trend"]}\n'
        if 'seasonality' in vision_analysis:
            prompt += f'- **Seasonality**: {vision_analysis["seasonality"]}\n'
        if 'volatility' in vision_analysis:
            prompt += f'- **Volatility**: {vision_analysis["volatility"]}\n'
        if 'regime_shifts' in vision_analysis:
            prompt += f'- **Regime Shifts**: {vision_analysis["regime_shifts"]}\n'
        if 'anomalies' in vision_analysis and vision_analysis["anomalies"]:
            anomalies_str = ', '.join(vision_analysis["anomalies"])
            prompt += f'- **Anomalies**: {anomalies_str}\n'
        if 'correlation_with_covariates' in vision_analysis:
            prompt += f'- **Correlation with Covariates**: {vision_analysis["correlation_with_covariates"]}\n'
        if 'future_covariates' in vision_analysis:
            prompt += f'- **Future Covariates Analysis**: {vision_analysis["future_covariates"]}\n'
        if 'forecast_hints' in vision_analysis:
            prompt += f'- **Forecast Hints**: {vision_analysis["forecast_hints"]}\n'
        
        prompt += '\nPlease consider these insights when making your forecast.\n'
    
    prompt += f'\n\n**Task:**\n'
    prompt += f'Based on the historical data above'
    additional_info = []
    # if future_covariate_data is not None:
    #     additional_info.append('the future covariate values provided')
    if vision_analysis is not None:
        additional_info.append('the time-series analysis insights provided')
    if additional_info:
        prompt += ', ' + ', and '.join(additional_info) if len(additional_info) > 1 else ', ' + additional_info[0]
    prompt += f', please forecast the target variable ({original_attr}) for the next {pred_window} recorded hours.'
    prompt += ' Please note that some data may contain missing values, so handle them carefully.'
    prompt += f' Please provide the complete forecast for the next {pred_window} hours, remember to give me the complete data. '
    prompt += 'You must provide the complete data.You mustn\'t omit any content.'
    prompt +='And your final answer must follow the format'
    prompt+="""
    <answer>
        \n```\n
        ...
        \n```\n
        </answer>
    Please obey the format strictly. And you must give me the complete answer.
    """
    

    os.makedirs(f'Memory/cases/origin/EPF/prompt/{data_name}', exist_ok=True)
    with open(f'Memory/cases/origin/EPF/prompt/{data_name}/prompt_{attr}_{data_name}_{look_back}_{pred_window}_{number}.txt', 'w') as f:
        f.write(prompt)
    print(f"[Prompt] Saved to Memory/cases/origin/EPF/prompt/{data_name}/prompt_{attr}_{data_name}_{look_back}_{pred_window}_{number}.txt", flush=True)
    # print(f"[Prompt] Length: {len(prompt)} chars, {prompt.count('\\n')+1} lines, ~{max(1, len(prompt)//4)} tokens (rough estimate)", flush=True)

    # model=deepseek_api_output(api_key=api_key,temperature=temperature,top_p=top_p)
    model=deepseek_api_output(api_key=api_key,temperature=temperature,top_p=top_p)

    answer=[]
    os.makedirs(f'Memory/cases/origin/EPF/{data_name}', exist_ok=True)

    if os.path.exists(f'Memory/cases/origin/EPF/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json'):
        with open(f'Memory/cases/origin/EPF/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json', 'r') as f:
                answer=json.load(f)
        if len(answer)==1:
            print('This task has been done!')
            return
        else:
            len_answer=len(answer)
    else:
            len_answer=0

    for k in range(1):
        if k<len_answer:
             continue
        else:
            print(f'{k+1} times')
            start_time = time.time()
            print(f"[API] Sending request to DeepSeek (attempt {k+1}) ...", flush=True)
            print(f"[API] Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            while True:
                try:
                    reasoning,result=model(prompt)
                    elapsed = time.time() - start_time
                    print(f"[API] Received response (attempt {k+1}) in {elapsed:.2f}s", flush=True)
                    print(f"[API] Reasoning length: {len(reasoning) if isinstance(reasoning, str) else 0} chars", flush=True)
                    print(f"[API] Answer length: {len(result) if isinstance(result, str) else 0} chars", flush=True)
                    print(reasoning)
                    print(result)
                    break
                except Exception as exc:
                    print(f"[API] Call failed (attempt {k+1}), retrying ...", flush=True)
                    print(f"[Exception] {exc}", flush=True)
                    time.sleep(1)
                    reasoning='Error'
                    result='Error'
            
            answer.append({'index':k,'reasoning':reasoning,'answer':result})

            with open(f'Memory/cases/origin/EPF/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json', 'w') as f:    
                json.dump(answer, f,indent=4)

    print('All done!')
