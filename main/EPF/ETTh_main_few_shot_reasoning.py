import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import json
import os
import re
from utils.api_ouput import deepseek_api_output
from utils.metrics import MSE, MAE

meaning_dict = {'HUFL': 'High UseFul Load',
                'HULL': 'High UseLess Load',
                'MUFL': 'Middle UseFul Load',
                'MULL': 'Middle UseLess Load',
                'LUFL': 'Low UseFul Load',
                'LULL': 'Low UseLess Load',
                'OT': 'Oil Temperature'}


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


def retrieve_similar_examples(current_data, historical_examples, top_k=5):
    """
    基于时间序列特征相似性检索相似的样例
    """
    # 提取当前数据的特征
    current_features = extract_time_series_features(current_data)
    
    # 提取历史样例的特征
    historical_features = []
    for example in historical_examples:
        features = extract_time_series_features(example['data'])
        historical_features.append(features)
    
    historical_features = np.array(historical_features)
    
    # 标准化特征
    scaler = StandardScaler()
    current_features_scaled = scaler.fit_transform(current_features.reshape(1, -1))
    historical_features_scaled = scaler.transform(historical_features)
    
    # 计算余弦相似度
    similarities = cosine_similarity(current_features_scaled, historical_features_scaled)[0]
    
    # 获取最相似的top_k个样例
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    similar_examples = []
    for idx in top_indices:
        similar_examples.append({
            'example': historical_examples[idx],
            'similarity': similarities[idx],
            'index': idx
        })
    
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
        example_prompt = f"Example {i+1} (Quality Score: {example_info['overall_quality']:.3f}):\n"
        example_prompt += f"Input data (past 96 time points):\n"
        example_prompt += example['data_string']
        idx = example['index']
        if use_summary:
            with open(f'Memory/summary_outputs/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{idx}.json', 'r') as f:
                summary = json.load(f)
            example_prompt += f"\nSummary of the reasoning process:\n{summary['overall_summary']}"
        else:
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
    few_shot_prompt += "\nBased on these examples, please solve the following similar task:\n\n"
    few_shot_prompt += base_prompt
    
    return few_shot_prompt


def ETTh_main_few_shot_reasoning(data_name, attr, look_back, pred_window, number, api_key, 
                                 temperature=0.6, top_p=0.7, num_similar_examples=5, 
                                 num_few_shot_examples=3,use_summary=False):
    """
    主函数：实现相似样例检索 + 质量再排序 + few-shot提示构造的one-shot推理
    从Memory中存储的训练集结果中检索相似案例
    """
    
    # 加载数据
    data_dir = '/data/songliv/TS/datasets/Single-mode/' + data_name + '.csv'
    data = pd.read_csv(data_dir)
    # 取前16个月的数据
    data = data[12 * 30 * 24 + 4 * 30 * 24 - look_back : 12 * 30 * 24 + 8 * 30 * 24]
    date = data.loc[:, 'date'].to_numpy()
    attr_data = data.loc[:, attr].to_numpy()
    data = pd.DataFrame(date, columns=['date'])
    data[attr] = attr_data
    
    # 准备历史样例数据 - 从Memory中加载训练集结果
    data_lookback = []
    historical_examples = []
    
    # 从Memory中加载所有可用的训练集结果
    memory_dir = f'Memory/result/{data_name}'
    if not os.path.exists(memory_dir):
        print(f"Warning: Memory directory {memory_dir} not found!")
        return
    
    # 获取所有相关的结果文件
    pattern = f'result_{attr}_{data_name}_{look_back}_{pred_window}_*.json'
    result_files = []
    for file in os.listdir(memory_dir):
        if file.startswith(f'result_{attr}_{data_name}_{look_back}_{pred_window}_') and file.endswith('.json'):
            result_files.append(file)
    
    print(f"Found {len(result_files)} training examples in Memory")
    
    # 加载历史样例数据
    for file in result_files:
        try:
            # 提取样例索引
            index_str = file.split('_')[-1].replace('.json', '')
            example_index = int(index_str)
            
            # 跳过当前要预测的样例
            # if example_index == number:
            #     continue
            
            # 加载对应的数据
            if example_index < 118:  # 确保索引在有效范围内
                try:
                    example_data = data.iloc[example_index * look_back:(example_index + 1) * look_back]
                    
                    # 检查数据是否有效
                    if len(example_data) == 0 or example_data[attr].isna().all():
                        print(f"Skipping invalid data for index {example_index}")
                        continue
                    
                    # 加载训练结果
                    result_file = os.path.join(memory_dir, file)
                    forecast_string = ""
                    reasoning_string = ""
                    if os.path.exists(result_file):
                        try:
                            with open(result_file, 'r') as f:
                                results = json.load(f)
                            if results and len(results) > 0:
                                forecast_string = results[0].get('answer', '')
                                reasoning_string = results[0].get('reasoning', '')
                        except Exception as e:
                            print(f"Error loading {result_file}: {e}")
                            continue
                    
                    historical_examples.append({
                        'data': example_data[attr].values,
                        'data_string': example_data.to_string(index=False),
                        'forecast_string': forecast_string,
                        'reasoning_string': reasoning_string,
                        'index': example_index,
                        'result_file': result_file
                    })
                except Exception as e:
                    print(f"Error processing data for index {example_index}: {e}")
                    continue
                
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
    
    print(f"Loaded {len(historical_examples)} historical examples from Memory")
    
    # 获取当前要预测的数据
    try:
        current_data = data.iloc[number * look_back:(number + 1) * look_back]
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
    similar_examples = retrieve_similar_examples(current_values, historical_examples, 
                                                top_k=num_similar_examples)
    print(f"Found {len(similar_examples)} similar examples")
    
    # 2. 质量再排序
    print("Step 2: Reranking examples by quality...")
    reranked_examples = rerank_examples_by_quality(similar_examples)
    print(f"Reranked {len(reranked_examples)} examples")
    
    # 打印质量排序信息
    for i, ex in enumerate(reranked_examples[:num_few_shot_examples]):
        print(f"  Top {i+1}: Index {ex['index']}, Quality: {ex['overall_quality']:.3f} "
              f"(Similarity: {ex['similarity']:.3f}, Prediction: {ex['predicted_quality']:.3f}, "
              f"Reasoning: {ex['reasoning_quality']:.3f}, Reasonableness: {ex['prediction_reasonableness']:.3f})")
    
    # 3. 构造基础提示
    base_prompt = f'You are an expert in time series forecasting.\n'
    base_prompt += f'The dataset records oil temperature and load metrics from electricity transformers, tracked between July 2016 and July 2018. '
    base_prompt += f'It is subdivided into four mini-datasets, with data sampled either hourly or every 15 minutes. '
    base_prompt += f'Here is the {meaning_dict[attr]} data of the transformer.\n'
    base_prompt += f'I will now give you data for the past {look_back} recorded dates, and please help me forecast the data for next {pred_window} recorded dates.\n'
    base_prompt += f'But please note that these data will have missing values, so be aware of that.\n'
    base_prompt += f'Please give me the complete data for the next {pred_window} recorded dates, remember to give me the complete data.\n'
    base_prompt += f'You must provide the complete data. You mustn\'t omit any content.\n'
    base_prompt += f'The data is as follows:\n'
    base_prompt += current_data.to_string(index=False)
    base_prompt += f'\nAnd your final answer must follow the format:\n'
    base_prompt += """
    <answer>
        \\n```\\n
        ...
        \\n```\\n
        </answer>
    Please obey the format strictly. And you must give me the complete answer.
    """
    
    # 4. 构造few-shot提示
    print("Step 3: Constructing few-shot prompt...")
    few_shot_prompt = construct_few_shot_prompt(base_prompt, reranked_examples, 
                                                num_examples=num_few_shot_examples,use_summary=use_summary,attr=attr,data_name=data_name,look_back=look_back,pred_window=pred_window)
    
    # 保存提示
    if not os.path.exists(f'output_{num_few_shot_examples}/prompt/{data_name}'):
        os.makedirs(f'output_{num_few_shot_examples}/prompt/{data_name}')
    with open(f'output_{num_few_shot_examples}/prompt/{data_name}/prompt_{attr}_{data_name}_{look_back}_{pred_window}_{number}.txt', 'w') as f:
        f.write(few_shot_prompt)
    
    # 5. 调用模型进行推理
    print("Step 4: Running few-shot reasoning...")
    model = deepseek_api_output(api_key=api_key, temperature=temperature, top_p=top_p)
    
    answer = []
    if not os.path.exists(f'output_{num_few_shot_examples}/result/{data_name}'):
        os.makedirs(f'output_{num_few_shot_examples}/result/{data_name}')
    
    result_file = f'output_{num_few_shot_examples}/result/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json'
    
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            answer = json.load(f)
        if len(answer) == 1:
            print('This task has been done!')
            return
        else:
            len_answer = len(answer)
    else:
        len_answer = 0
    
    for k in range(1):
        if k < len_answer:
            continue
        else:
            print(f'{k+1} times')
            while True:
                try:
                    reasoning, result = model(few_shot_prompt)
                    print("Reasoning:", reasoning[:200] + "..." if len(reasoning) > 200 else reasoning)
                    print("Result:", result[:200] + "..." if len(result) > 200 else result)
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    reasoning = 'Error'
                    result = 'Error'
            
            # 保存结果，包含检索和排序信息
            answer.append({
                'index': k,
                'reasoning': reasoning,
                'answer': result,
                'similar_examples_used': [
                    {
                        'index': ex['index'],
                        'similarity': ex['similarity'],
                        'overall_quality': ex['overall_quality']
                    } for ex in reranked_examples[:num_few_shot_examples]
                ]
            })
            
            with open(result_file, 'w') as f:
                json.dump(answer, f, indent=4)
    
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
