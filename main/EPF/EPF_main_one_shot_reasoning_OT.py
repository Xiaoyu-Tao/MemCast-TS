import pandas as pd
# from utils.api_ouput import qianduoduo_api_output
# from utils.api_ouput import nvidia_api_output
from utils.api_ouput import deepseek_api_output,gpt4_api_output
import json
import os

meaning_dict = {
    'Grid load forecast': 'Forecasted grid electricity load in the Nord Pool system (MW)',
    'Wind power forecast': 'Forecasted wind power generation in the Nord Pool system (MW)',
    'OT': 'Observed Nord Pool electricity price (€/MWh, 1-hour resolution)'
}



def EPF_main_one_shot_reasoning_OT(data_name,attr,look_back,pred_window,number,api_key,temperature,top_p):

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
    
    # 获取目标变量的实际列名
    if attr not in data.columns:
        # 尝试查找匹配的列
        matching_cols = [col for col in data.columns if attr.lower().strip() in col.lower().strip() or col.lower().strip() in attr.lower().strip()]
        if matching_cols:
            attr_col = matching_cols[0]
        else:
            attr_col = attr
    else:
        attr_col = attr
    
    # 构建只包含目标变量的数据框
    data_full = pd.DataFrame(data['date'].values, columns=['date'])
    if attr_col in data.columns:
        data_full[original_attr] = data[attr_col].values
    
    # 调试信息：打印实际包含的列
    print(f"Target variable: {original_attr} (column: {attr_col})")
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

    # 构建目标变量数据框
    target_data = pd.DataFrame({
        'date': data_lookback[number]['date'],
        original_attr: data_lookback[number][original_attr]
    })

    prompt = ''
    prompt += ' The dataset comes from the Nord Pool electricity market and contains hourly records of observed electricity prices.'
    prompt += f'\n\n**Target Variable to Forecast:**\n'
    prompt += f'Here is the historical data of {meaning_dict.get(original_attr, original_attr)} (the variable you need to forecast) for the past {look_back} recorded hours:\n'
    prompt += target_data.to_string(index=False)
    
    prompt += f'\n\n**Task:**\n'
    prompt += f'Based on the historical data above, please forecast the target variable ({original_attr}) for the next {pred_window} recorded hours.'
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
    

    if not os.path.exists(f'output_OT/prompt/{data_name}'):
        os.makedirs(f'output_OT/prompt/{data_name}')
    with open(f'output_OT/prompt/{data_name}/prompt_{attr}_{data_name}_{look_back}_{pred_window}_{number}.txt', 'w') as f:
        f.write(prompt)

    model=gpt4_api_output(api_key=api_key,temperature=temperature,top_p=top_p)

    answer=[]
    if not os.path.exists(f'output_OT/result/{data_name}'):
        os.makedirs(f'output_OT/result/{data_name}')

    if os.path.exists(f'output_OT/result/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json'):
        with open(f'output_OT/result/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json', 'r') as f:
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

            with open(f'output_OT/result/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json', 'w') as f:    
                json.dump(answer, f,indent=4)

    print('All done!')