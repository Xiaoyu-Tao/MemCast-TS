import pandas as pd
# from utils.api_ouput import qianduoduo_api_output
# from utils.api_ouput import nvidia_api_output
from utils.api_ouput import deepseek_multimodal_api_output
import json
import os

meaning_dict = {
    'Grid load forecast': 'Forecasted grid electricity load in the Nord Pool system (MW)',
    'Wind power forecast': 'Forecasted wind power generation in the Nord Pool system (MW)',
    'OT': 'Observed Nord Pool electricity price (€/MWh, 1-hour resolution)'
}



def EPF_main_one_shot_reasoning(data_name,attr,look_back,pred_window,number,api_key,temperature,top_p):

    # Construct image path from visualization output
    image_path = f'/data/songliv/TS/TimeReasoner/output/EPF/NP/NP_visualization_{number:03d}.png'
    
    if not os.path.exists(image_path):
        print(f'Warning: Image not found at {image_path}. Please ensure visualizations are generated first.')
        return

    prompt = """You are an expert in time-series analysis and forecasting. 
The following image shows a visualization from the Nord Pool electricity market, where:
- The blue line represents electricity price (€/MWh)
- The solid line represents grid load forecast
- The dashed orange line represents wind power forecast
- The shaded region indicates the forecast horizon (future window)
- Local maxima and minima are marked by triangles.

Please carefully analyze this time-series visualization and summarize the following aspects in structured JSON format:

{
  "trend": "overall upward / downward / oscillating",
  "seasonality": "daily / weekly / none",
  "volatility": "high / moderate / low",
  "regime_shifts": "describe any significant pattern changes, e.g., sudden drops or peaks",
  "anomalies": ["timestamp1", "timestamp2", ...]  // any sharp outliers
  "correlation_with_covariates": "describe how price changes relate to load and wind",
  "future_covariates": "describe the future conditions shown (load/wind trend)",
  "forecast_hints": "short textual hints on how to forecast the next 24 hours"
}

Your answer must be in pure JSON format without any additional commentary."""

    if not os.path.exists(f'output/prompt/{data_name}'):
        os.makedirs(f'output/prompt/{data_name}')
    with open(f'output/prompt/{data_name}/prompt_{attr}_{data_name}_{look_back}_{pred_window}_{number}.txt', 'w') as f:
        f.write(prompt)

    model=deepseek_multimodal_api_output(api_key=api_key,temperature=temperature,top_p=top_p)

    answer=[]
    if not os.path.exists(f'output/result/{data_name}'):
        os.makedirs(f'output/result/{data_name}')

    if os.path.exists(f'output/result/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json'):
        with open(f'output/result/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json', 'r') as f:
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
                    result=model(prompt, image_path=image_path)
                    # print(reasoning)
                    print(result)
                    break
                except:
                    result='Error'
            
            answer.append({'index':k,'answer':result})

            with open(f'output/result/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json', 'w') as f:
                json.dump(answer, f,indent=4)

    print('All done!')


# 别名函数，保持兼容性
def EPF_main_vison_to_text(data_name, attr, look_back, pred_window, number, api_key, temperature, top_p):
    """别名函数，调用 EPF_main_one_shot_reasoning"""
    return EPF_main_one_shot_reasoning(data_name, attr, look_back, pred_window, number, api_key, temperature, top_p)