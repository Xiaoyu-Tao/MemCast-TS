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

    data_dir = '/data/songliv/TS/datasets/EPF/' + data_name + '.csv'
    data = pd.read_csv(data_dir)
    # 清理列名，去除可能的空格
    data.columns = data.columns.str.strip()
    
    # 取前12个月的数据 (假设每小时一个数据点，12个月 = 12 * 30 * 24)
    # end_idx = int(len(data) * 0.8) + look_back
    # if end_idx > len(data):
    #     end_idx = len(data)
    # data = data[:end_idx]
    start_idx = int(len(data) * 0.8) - look_back
    if start_idx < 0:
        start_idx = 0
    data = data[start_idx:]
    
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
    
    # 获取另外两个变量作为协变量
    covariates = {}
    for key, col_name in all_attrs_dict.items():
        if key != original_attr and col_name in data.columns:
            covariates[key] = col_name
    
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
    for i in range(min(60, max_samples)):
        start_idx = i * slide_window
        end_idx = start_idx + look_back
        if end_idx <= len(data_full):
            data_lookback.append(data_full.iloc[start_idx:end_idx])

    # 计算当前窗口的历史数据结束位置
    current_start_idx = number * slide_window
    current_end_idx = current_start_idx + look_back
    
    # 分别构建目标变量和协变量的数据框
    target_data = pd.DataFrame({
        'date': data_lookback[number]['date'],
        original_attr: data_lookback[number][original_attr]
    })
    
    # 构建协变量数据框（如果有协变量）
    covariate_data = None
    if covariates:
        cov_cols = ['date'] + list(covariates.keys())
        covariate_data = data_lookback[number][cov_cols].copy()
    
    # 提取未来的协变量数据
    future_covariate_data = None
    if covariates and current_end_idx + pred_window <= len(data_full):
        future_start_idx = current_end_idx
        future_end_idx = current_end_idx + pred_window
        future_cov_cols = ['date'] + list(covariates.keys())
        future_covariate_data = data_full.iloc[future_start_idx:future_end_idx][future_cov_cols].copy()

    # 尝试加载视觉分析的推理结果（如果存在）
    vision_analysis = None
    vision_result_path = f'output/result/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json'
    if os.path.exists(vision_result_path):
        try:
            with open(vision_result_path, 'r') as f:
                vision_results = json.load(f)
                if vision_results and len(vision_results) > 0:
                    vision_answer_str = vision_results[0].get('answer', '')
                    if vision_answer_str:
                        # answer字段本身是一个JSON字符串，需要再次解析
                        vision_analysis = json.loads(vision_answer_str)
                        print(f"Loaded vision analysis: {list(vision_analysis.keys())}")
        except Exception as e:
            print(f"Warning: Could not load vision analysis from {vision_result_path}: {e}")
            vision_analysis = None
    else:
        print(f"Vision analysis file not found: {vision_result_path}")

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
        prompt += f'These covariates show the relationships between electricity demand (grid load), renewable supply (wind power), and price, which can inform your forecast:\n'
        prompt += '\n'.join(cov_meanings_list)
        prompt += f'\n\nHistorical data of covariates for the past {look_back} recorded hours:\n'
        prompt += covariate_data.to_string(index=False)
        
        # # 添加未来的协变量数据
        # if future_covariate_data is not None:
        #     prompt += f'\n\n**Future Covariates (Known Future Values):**\n'
        #     prompt += f'I also provide the future values of the covariates for the next {pred_window} recorded hours. '
        #     prompt += f'These are known forecasted values that you can use to help predict the target variable:\n'
        #     prompt += future_covariate_data.to_string(index=False)
    
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
    

    if not os.path.exists(f'output_see/prompt/{data_name}'):
        os.makedirs(f'output_see/prompt/{data_name}')
    with open(f'output_see/prompt/{data_name}/prompt_{attr}_{data_name}_{look_back}_{pred_window}_{number}.txt', 'w') as f:
        f.write(prompt)

    model=gpt4_api_output(api_key=api_key,temperature=temperature,top_p=top_p)

    answer=[]
    if not os.path.exists(f'output_see/result/{data_name}'):
        os.makedirs(f'output_see/result/{data_name}')

    if os.path.exists(f'output_see/result/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json'):
        with open(f'output_see/result/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json', 'r') as f:
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
            while True:
                try:
                    reasoning,result=model(prompt)
                    print(reasoning)
                    print(result)
                    break
                except:
                    reasoning='Error'
                    result='Error'
            
            answer.append({'index':k,'reasoning':reasoning,'answer':result})

            with open(f'output_see/result/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json', 'w') as f:    
                json.dump(answer, f,indent=4)

    print('All done!')