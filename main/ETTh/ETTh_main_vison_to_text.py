import pandas as pd
from utils.api_ouput import deepseek_multimodal_api_output
import json
import os

meaning_dict = {
    'HUFL': 'High Useful Load (high-frequency component of useful electrical load)',
    'HULL': 'High UnUseful Load (high-frequency component of non-useful electrical load)',
    'MUFL': 'Medium Useful Load (medium-frequency component of useful electrical load)',
    'MULL': 'Medium UnUseful Load (medium-frequency component of non-useful electrical load)',
    'LUFL': 'Low Useful Load (low-frequency component of useful electrical load)',
    'LULL': 'Low UnUseful Load (low-frequency component of non-useful electrical load)',
    'OT': 'Oil Temperature (transformer oil temperature, prediction target)'
}



def ETTh_main_one_shot_reasoning(data_name, attr, look_back, pred_window, number, api_key, temperature, top_p, mode='memory'):

    image_path = f'./Memory/visual/image/ETTh/{data_name}/{mode}/{data_name}_visualization_{number:03d}.png'
    meta_path = image_path.replace('.png', '.json')

    if not os.path.exists(image_path):
        print(f'Warning: Image not found at {image_path}. Please ensure visualizations are generated first.')
        return

    target_line_text = "- The blue line represents the target variable Oil Temperature"
    covariates_line_text = "- The secondary axis overlays two covariates (Load variants); one line is solid and the other is dashed"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            covs = meta.get('covariates', [])
            if isinstance(covs, list) and len(covs) > 0:
                cov_details = []
                for c in covs[:2]:
                    cov_details.append(f"{c} ({meaning_dict.get(c, c)})")
                covariates_line_text = f"""- The solid line represents {cov_details[0]}
- The dashed orange line represents {cov_details[1]}."""
        except:
            pass

    prompt = f"""You are an expert in time-series analysis and forecasting.
The following image shows a visualization from the Electricity Transformer Temperature, hourly(ETTh), where:
{target_line_text}
{covariates_line_text}
- The shaded region indicates the forecast horizon (future window)
- Local maxima and minima are marked by triangles.

Please carefully analyze this time-series visualization and summarize the following aspects in structured JSON format:

{{
  "trend": "overall upward / downward / oscillating",
  "seasonality": "daily / weekly / none",
  "volatility": "high / moderate / low",
  "regime_shifts": "describe any significant pattern changes, e.g., sudden drops or peaks",
  "anomalies": ["timestamp1", "timestamp2", ...]  // any sharp outliers
  "correlation_with_covariates": "describe how the target relates to the two covariates",
  "future_covariates": "describe the covariate behavior in the shaded forecast horizon",
  "forecast_hints": "short textual hints on how to forecast the next 24 hours (or the specified horizon)"
}}

Your answer must be in pure JSON format without any additional commentary."""

    os.makedirs(f'Memory/visual/prompt/ETTh/{data_name}', exist_ok=True)
    with open(f'Memory/visual/prompt/ETTh/{data_name}/prompt_{attr}_{data_name}_{look_back}_{pred_window}_{number}.txt', 'w') as f:
        f.write(prompt)

    model = deepseek_multimodal_api_output(api_key=api_key, temperature=temperature, top_p=top_p)

    answer = []
    result_dir = f'Memory/visual/analysis/ETTh/{data_name}/{mode}'
    os.makedirs(result_dir, exist_ok=True)
    result_path = f'{result_dir}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json'

    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
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
                    result = model(prompt, image_path=image_path)
                    print(result)
                    break
                except:
                    result = 'Error'

            answer.append({'index': k, 'answer': result})

            with open(result_path, 'w') as f:
                json.dump(answer, f, indent=4)

    print('All done!')


# 别名函数，保持兼容性
def ETTh_main_vison_to_text(data_name, attr, look_back, pred_window, number, api_key, temperature, top_p, mode='memory'):
    """别名函数，调用 ETTh_main_one_shot_reasoning"""
    return ETTh_main_one_shot_reasoning(data_name, attr, look_back, pred_window, number, api_key, temperature, top_p, mode=mode)
