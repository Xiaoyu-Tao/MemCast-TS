import pandas as pd
import numpy as np
from utils.api_ouput import deepseek_api_output
import json
import os

meaning_dict = {
    'HUFL': 'High UseFul Load',
    'HULL': 'High UseLess Load',
    'MUFL': 'Middle UseFul Load',
    'MULL': 'Middle UseLess Load',
    'LUFL': 'Low UseFul Load',
    'LULL': 'Low UseLess Load',
    'OT': 'Oil Temperature'
}


def get_desc(domain, seq_len, pred_len):
    """Generate description for the dataset domain"""
    descriptions = {
        'ETTh1': f'This is Electricity Transformer Temperature (ETTh1) data, recorded hourly. We use {seq_len} historical observations to predict the next {pred_len} values.',
        'ETTh2': f'This is Electricity Transformer Temperature (ETTh2) data, recorded hourly. We use {seq_len} historical observations to predict the next {pred_len} values.',
        'ETTm1': f'This is Electricity Transformer Temperature (ETTm1) data, recorded every 15 minutes. We use {seq_len} historical observations to predict the next {pred_len} values.',
        'ETTm2': f'This is Electricity Transformer Temperature (ETTm2) data, recorded every 15 minutes. We use {seq_len} historical observations to predict the next {pred_len} values.',
    }
    return descriptions.get(domain, f'Time series data with {seq_len} historical observations to predict {pred_len} future values.')


def multimodal_main_one_shot_reasoning(data_name, attr, look_back, pred_window, number, 
                                       api_key, temperature, top_p, 
                                       root_path='/data/songliv/TS/datasets/time-mmd',
                                       text_len=1, use_textual=True, set_type='test'):
    """
    Multi-modal one-shot reasoning for time series forecasting (time-mmd format)
    
    This function follows the same data structure as Dataset_Custom class:
    - numerical/{data_name}.csv: numerical time series data
    - textual/{data_name}_report.csv: textual reports with start_date, end_date, fact
    - textual/{data_name}_search.csv: textual search data (optional)
    
    Args:
        data_name: name of the dataset (e.g., 'ETTh1')
        attr: target attribute to forecast (e.g., 'OT', 'HUFL')
        look_back: number of historical time steps to use (seq_len)
        pred_window: number of future time steps to predict (pred_len)
        number: which segment of the data to use (0-9)
        api_key: API key for the model
        temperature: temperature parameter for generation
        top_p: top_p parameter for generation
        root_path: root path to the time-mmd dataset
        text_len: length of text context to use (number of time steps)
        use_textual: whether to include textual data in the prompt
        set_type: 'train', 'valid', or 'test' (default: 'test')
    """
    
    # Read numerical data (following Dataset_Custom.__read_data__)
    num_data_dir = os.path.join(root_path, 'numerical', f'{data_name}.csv')
    
    if not os.path.exists(num_data_dir):
        print(f"Error: Numerical data not found at {num_data_dir}")
        return
    
    df_num = pd.read_csv(num_data_dir)
    
    # Read textual data if available and requested
    df_report = None
    df_search = None
    if use_textual:
        try:
            report_path = os.path.join(root_path, 'textual', f'{data_name}_report.csv')
            search_path = os.path.join(root_path, 'textual', f'{data_name}_search.csv')
            
            if os.path.exists(report_path):
                df_report = pd.read_csv(report_path)
                df_report = df_report.dropna(axis='index', how='any', subset=['fact'])
                df_report['start_date'] = pd.to_datetime(df_report['start_date'])
                df_report['end_date'] = pd.to_datetime(df_report['end_date'])
                df_report = df_report.sort_values('start_date', ascending=True).reset_index(drop=True)
                print(f"Loaded textual report data: {len(df_report)} records")
            
            if os.path.exists(search_path):
                df_search = pd.read_csv(search_path)
                df_search = df_search.dropna(axis='index', how='any', subset=['fact'])
                df_search['start_date'] = pd.to_datetime(df_search['start_date'])
                df_search['end_date'] = pd.to_datetime(df_search['end_date'])
                df_search = df_search.sort_values('start_date', ascending=True).reset_index(drop=True)
                print(f"Loaded textual search data: {len(df_search)} records")
        except Exception as e:
            print(f"Warning: Could not load textual data: {e}")
            use_textual = False
    
    # Process numerical data (following Dataset_Custom.__read_data__)
    df_num = df_num.dropna(axis='index', how='any', subset=[attr])
    df_num['date'] = pd.to_datetime(df_num['date'])
    
    # Check if start_date and end_date columns exist
    has_date_range = 'start_date' in df_num.columns and 'end_date' in df_num.columns
    if has_date_range:
        df_num['start_date'] = pd.to_datetime(df_num['start_date'])
        df_num['end_date'] = pd.to_datetime(df_num['end_date'])
    
    df_num = df_num.sort_values('date', ascending=True).reset_index(drop=True)
    
    # Calculate data splits (exactly following Dataset_Custom)
    num_train = int(len(df_num) * 0.7)
    num_test = int(len(df_num) * 0.2)
    num_vali = len(df_num) - num_train - num_test
    
    # Define borders (following Dataset_Custom)
    border1s = [0, num_train - look_back, len(df_num) - num_test - look_back]
    border2s = [num_train, num_train + num_vali, len(df_num)]
    
    # Select the appropriate split
    type_map = {'train': 0, 'valid': 1, 'test': 2}
    set_type_idx = type_map.get(set_type, 2)
    border1 = border1s[set_type_idx]
    border2 = border2s[set_type_idx]
    
    print(f"Using {set_type} set: border1={border1}, border2={border2}, total_length={border2-border1}")
    
    # Extract the data segment for this split
    df_segment = df_num[border1:border2].reset_index(drop=True)
    
    # Create data lookback segments
    max_segments = (len(df_segment) - look_back) // look_back
    if max_segments <= 0:
        print(f"Error: Not enough data for even 1 segment. Need at least {look_back} samples.")
        return
    
    data_lookback = []
    for i in range(min(10, max_segments)):
        start_idx = i * look_back
        end_idx = (i + 1) * look_back
        if end_idx <= len(df_segment):
            segment = df_segment.iloc[start_idx:end_idx]
            data_lookback.append(segment)
    
    if number >= len(data_lookback):
        print(f"Error: number {number} is out of range. Only {len(data_lookback)} segments available.")
        return
    
    # Get the specific segment (this is seq_x in Dataset_Custom)
    segment = data_lookback[number]
    
    # Collect textual information for this segment (following Dataset_Custom.collect_text)
    text_info = ""
    text_mark = 0
    
    if use_textual and df_report is not None and has_date_range:
        # Get date range for text context (following Dataset_Custom.__getitem__)
        # text_begin corresponds to s_end - text_len in Dataset_Custom
        text_begin_idx = max(0, len(segment) - text_len)
        text_end_idx = len(segment) - 1
        
        # Get the actual segment indices in the full data
        segment_start_in_full = border1 + number * look_back + text_begin_idx
        segment_end_in_full = border1 + number * look_back + text_end_idx
        
        context_start_date = df_num.iloc[segment_start_in_full]['start_date']
        context_end_date = df_num.iloc[segment_end_in_full]['end_date']
        
        # Collect relevant reports (following Dataset_Custom.collect_text)
        relevant_reports = df_report.loc[
            (df_report.end_date >= context_start_date) & 
            (df_report.end_date <= context_end_date)
        ]
        
        if not relevant_reports.empty:
            text_mark = 1
            # Format text with date markers (following Dataset_Custom.add_datemark)
            report_list = []
            desc = get_desc(data_name, look_back, pred_window)
            report_list.append(desc)
            
            for _, row in relevant_reports.iterrows():
                date_str = f"{row['start_date'].strftime('%Y-%m-%d')} to {row['end_date'].strftime('%Y-%m-%d')}"
                report_list.append(f"{date_str}: {row['fact']}")
            
            text_info = "\n\nAdditional contextual information:\n" + '\n'.join(report_list) + "\n"
            print(f"Found {len(relevant_reports)} relevant textual reports")
        else:
            text_info = ""
            print("No relevant textual reports found for this time period")
    
    # Build the prompt
    prompt = 'You are an expert in time series forecasting. '
    
    if attr in meaning_dict:
        prompt += f'Here is the {meaning_dict[attr]} data of the transformer. '
    else:
        prompt += f'Here is the {attr} data. '
    
    prompt += f'I will now give you data for the past {look_back} recorded dates, and please help me forecast the data for next {pred_window} recorded dates. '
    prompt += 'But please note that these data may have missing values, so be aware of that. '
    prompt += f'Please give me the complete data for the next {pred_window} recorded dates, remember to give me the complete data. '
    prompt += 'You must provide the complete data. You mustn\'t omit any content. '
    
    # Add textual context if available
    if text_info:
        prompt += text_info
        prompt += '\n\n'
    
    prompt += 'Think step by step: analyze trend, seasonality, and any contextual information provided, then forecast. '
    prompt += 'The numerical data is as follows:\n'
    
    # Format the data
    if 'date' in segment.columns:
        data_to_show = segment[['date', attr]]
    else:
        data_to_show = segment[[attr]]
    
    prompt += data_to_show.to_string(index=False)
    
    prompt += '\n\nAnd your final answer must follow the format:\n'
    prompt += """
    <answer>
        \n```\n
        ...
        \n```\n
    </answer>
    Please obey the format strictly. And you must give me the complete answer.
    """
    
    # Save prompt
    output_dir = f'output/prompt/{data_name}_multimodal'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    prompt_filename = f'prompt_{attr}_{data_name}_{look_back}_{pred_window}_{number}.txt'
    with open(os.path.join(output_dir, prompt_filename), 'w') as f:
        f.write(prompt)
    
    # Initialize model
    model = deepseek_api_output(api_key=api_key, temperature=temperature, top_p=top_p)
    
    # Check if results already exist
    result_dir = f'output/result/{data_name}_multimodal'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    result_filename = f'result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json'
    result_path = os.path.join(result_dir, result_filename)
    
    answer = []
    len_answer = 0
    
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            answer = json.load(f)
        if len(answer) == 3:
            print('This task has been done!')
            return
        else:
            len_answer = len(answer)
    
    # Generate 3 responses
    for k in range(3):
        if k < len_answer:
            continue
        else:
            print(f'{k+1} times')
            while True:
                try:
                    reasoning, result = model(prompt)
                    print(f"Reasoning:\n{reasoning}\n")
                    print(f"Result:\n{result}\n")
                    break
                except Exception as e:
                    print(f"Error occurred: {e}, retrying...")
                    reasoning = 'Error'
                    result = 'Error'
            
            answer.append({
                'index': k,
                'reasoning': reasoning,
                'answer': result,
                'used_textual': use_textual
            })
            
            with open(result_path, 'w') as f:
                json.dump(answer, f, indent=4)
    
    print('All done!')


if __name__ == "__main__":
    # Example usage
    multimodal_main_one_shot_reasoning(
        data_name='ETTh1',
        attr='OT',
        look_back=96,
        pred_window=96,
        number=0,
        api_key='your_api_key_here',
        temperature=0.6,
        top_p=0.7,
        root_path='/data/songliv/TS/datasets',
        text_len=1,
        use_textual=True
    )

