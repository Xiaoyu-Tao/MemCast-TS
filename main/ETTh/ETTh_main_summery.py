from utils.api_ouput import deepseek_api_output
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

def load_memory_samples(data_name: str, attr: str = None, look_back: int = None, pred_window: int = None, number: int = None) -> List[Dict[str, Any]]:
    """
    从Memory目录中加载样本数据
    
    Args:
        data_name: 数据集名称，如'ETTh1'
        attr: 属性名称，如'HUFL'，可选
        look_back: 回看窗口大小，可选
        pred_window: 预测窗口大小，可选
    
    Returns:
        List[Dict]: 包含样本数据的列表，每个字典包含index, reasoning, answer等字段
    """
    memory_dir = f'Memory/result/{data_name}'
    if not os.path.exists(memory_dir):
        print(f"Warning: Memory directory {memory_dir} not found!")
        return []
    
    samples = []
    result_files = []
    
    # 获取所有相关的结果文件
    for file in os.listdir(memory_dir):
        if file.endswith('.json'):
            if attr and look_back and pred_window:
                # 如果指定了具体参数，只匹配对应的文件
                pattern = f'result_{attr}_{data_name}_{look_back}_{pred_window}_{number}.json'
                if file.startswith(pattern):
                    result_files.append(file)
            else:
                # 否则加载所有文件
                if file.startswith('result_'):
                    result_files.append(file)
    
    print(f"Found {len(result_files)} result files in Memory")
    
    # 加载每个文件中的数据
    for file in result_files:
        try:
            file_path = os.path.join(memory_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            
            # 提取文件信息
            parts = file.replace('.json', '').split('_')
            if len(parts) >= 6:
                file_attr = parts[1]
                file_look_back = int(parts[3])
                file_pred_window = int(parts[4])
                file_index = int(parts[5])
            else:
                continue
            
            # 处理文件中的每个样本
            for sample in file_data:
                if isinstance(sample, dict):
                    sample_info = {
                        'file_name': file,
                        'file_attr': file_attr,
                        'file_look_back': file_look_back,
                        'file_pred_window': file_pred_window,
                        'file_index': file_index,
                        'sample_index': sample.get('index', 0),
                        'reasoning': sample.get('reasoning', ''),
                        'answer': sample.get('answer', ''),
                        'similar_examples_used': sample.get('similar_examples_used', []),
                        'overall_quality': sample.get('overall_quality', None),
                        'predicted_quality': sample.get('predicted_quality', None)
                    }
                    samples.append(sample_info)
                    
        except Exception as e:
            print(f"Error loading file {file}: {e}")
            continue
    
    print(f"Loaded {len(samples)} samples from Memory")
    return samples


def summarize_reasoning_with_deepseek(samples: List[Dict[str, Any]], api_key: str, temperature: float = 0.7, top_p: float = 1.0) -> Dict[str, Any]:
    """
    使用 DeepSeek API 对思考轨迹进行总结分析。
    
    Args:
        samples: 包含多个样本思考轨迹的列表。
        api_key: 你的 DeepSeek API 密钥。
        temperature: 控制生成文本的随机性。
        top_p: 控制核心采样的参数。
        
    Returns:
        一个包含结构化总结的字典。
    """
    if not samples:
        print("No samples to analyze!")
        return {}

    print(f"Analyzing {len(samples)} reasoning trajectories with DeepSeek API...")

    # 提取所有非空的思考轨迹
    reasoning_texts = [sample.get('reasoning', '') for sample in samples]
    reasoning_texts = [text for text in reasoning_texts if text and text.strip()]
    
    if not reasoning_texts:
        print("No valid reasoning texts found in samples.")
        return {}

    # 将所有轨迹拼接成一个大的文本块，并使用分隔符
    full_reasoning_text = "\n\n--- Sample Separator ---\n\n".join(reasoning_texts)

    # 构建发送给 DeepSeek 的 Prompt
    system_prompt = """You are an expert AI analyst specializing in time-series forecasting. Your task is to analyze a collection of reasoning trajectories from a forecasting model. These trajectories explain the step-by-step thinking process for predicting future data points.

Carefully review all the provided reasoning samples and generate a concise, structured summary in JSON format. The summary should highlight common patterns, methods, and insights across all samples."""

    # 限制输入文本长度以避免超出API限制
    user_prompt = f"""Here are the reasoning trajectories from multiple time-series forecasting tasks. Please analyze them and provide a summary.

**Reasoning Trajectories:**
```
{full_reasoning_text[:15000]} 
```

**Your Task:**
Based on the trajectories, generate a JSON object with the following structure:
1.  `overall_summary`: A brief, high-level paragraph summarizing the general approach and quality of the reasoning.
2.  `method_distribution`: A dictionary where keys are the primary forecasting methods identified (e.g., "SARIMA/ARIMA", "Exponential Smoothing", "Heuristic Analysis", "Statistical Analysis", "Simple Averaging") and values are the approximate percentage of samples using that method (e.g., "40%").
3.  `common_data_patterns`: A list of the most frequently identified patterns in the time-series data (e.g., "Daily seasonality", "Upward trend", "Handling of missing values", "Outlier detection").
4.  `key_insights`: A list of 3-5 of the most important or recurring insights or analytical steps mentioned across the samples.
5.  `complexity_analysis`: A brief description of the overall complexity of the reasoning processes (e.g., "Mostly simple heuristics", "Moderately complex with statistical considerations", "Highly detailed with multi-step analysis").

Please provide only the JSON object in your response, without any introductory text or code block formatting.
"""

    # 调用 DeepSeek API
    try:
        model = deepseek_api_output(api_key=api_key, temperature=temperature, top_p=top_p)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        print("Sending request to DeepSeek API...")
        _, summary_str = model(full_prompt)
        print("Received response from DeepSeek API.")
        
        # 清理和解析返回的JSON
        if '```json' in summary_str:
            summary_str = summary_str.split('```json\n')[1].split('\n```')[0]
        elif '```' in summary_str:
            summary_str = summary_str.replace('```', '')
            
        summary_json = json.loads(summary_str.strip())
        summary_json['analysis_timestamp'] = datetime.now().isoformat()
        summary_json['total_samples_analyzed'] = len(reasoning_texts)
        
        return summary_json
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from DeepSeek API response. Error: {e}")
        print(f"Raw response: {summary_str}")
        return {"error": "JSONDecodeError", "raw_response": summary_str}
    except Exception as e:
        print(f"An error occurred while calling the DeepSeek API: {e}")
        return {"error": str(e)}


def print_deepseek_summary(summary: Dict[str, Any]):
    """
    以易于阅读的格式打印由 DeepSeek 生成的总结报告。
    
    Args:
        summary: 从 DeepSeek API 返回的总结字典。
    """
    if not summary or 'overall_summary' not in summary:
        print("Invalid or empty summary provided.")
        if summary.get("error"):
            print(f"An error occurred during analysis: {summary['error']}")
        return
        
    print("\n" + "="*80)
    print("REASONING TRAJECTORY SUMMARY (Generated by DeepSeek)")
    print("="*80)
    
    print(f"\nTotal Samples Analyzed: {summary.get('total_samples_analyzed', 'N/A')}")
    print(f"Analysis Timestamp: {summary.get('analysis_timestamp', 'N/A')}")
    
    print("\n📜 OVERALL SUMMARY:")
    print(f"  {summary.get('overall_summary', 'Not available.')}")
    
    print(f"\n📊 METHOD DISTRIBUTION:")
    method_dist = summary.get('method_distribution', {})
    if method_dist:
        for method, percentage in method_dist.items():
            print(f"  - {method}: {percentage}")
    else:
        print("  Not available.")
        
    print(f"\n📈 COMMON DATA PATTERNS:")
    patterns = summary.get('common_data_patterns', [])
    if patterns:
        for pattern in patterns:
            print(f"  - {pattern}")
    else:
        print("  Not available.")
        
    print(f"\n💡 KEY INSIGHTS:")
    insights = summary.get('key_insights', [])
    if insights:
        for insight in insights:
            print(f"  - {insight}")
    else:
        print("  Not available.")
        
    print(f"\n🧠 COMPLEXITY ANALYSIS:")
    print(f"  {summary.get('complexity_analysis', 'Not available.')}")
    
    print("\n" + "="*80)


def analyze_memory_reasoning(data_name: str, api_key: str, attr: str = None, look_back: int = None,  
                             pred_window: int = None, output_file: str = None, output_dir: str = None,  
                             print_report: bool = True, temperature: float = 0.2, top_p: float = 1.0,
                             overwrite: bool = False, number: int = None) -> Dict[str, Any]:
    """
    分析Memory中思考轨迹的主函数 (使用 DeepSeek API)。

    Args:
        data_name: 数据集名称。
        api_key: DeepSeek API 密钥。
        attr: 属性名称，可选。
        look_back: 回看窗口大小，可选。
        pred_window: 预测窗口大小，可选。
        output_file: 输出文件路径，可选。
        print_report: 是否打印报告。
        temperature: 模型温度。
        top_p: Top-p采样。
        overwrite: 是否允许覆盖已存在的结果文件。
        number: 样本编号，可选。
    """
    print(f"Loading samples from Memory for {data_name}...")

    samples = load_memory_samples(data_name, attr, look_back, pred_window, number)
    if not samples:
        print("No samples found!")
        return {}

    # ==== [1] 准备输出路径并检查结果文件是否已存在 ====
    if output_dir and not output_file:
        try:
            target_dir = os.path.join(output_dir, data_name) if data_name else output_dir
            os.makedirs(target_dir, exist_ok=True)

            attr_str = attr if attr else 'ALL'
            lb_str = str(look_back) if look_back is not None else 'NA'
            pw_str = str(pred_window) if pred_window is not None else 'NA'
            file_name = f"result_{attr_str}_{data_name}_{lb_str}_{pw_str}_{number}.json"
            output_file = os.path.join(target_dir, file_name)
        except Exception as e:
            print(f"Error preparing output directory or filename: {e}")

    # ==== [2] 如果文件存在且未允许覆盖，直接跳过推理 ====
    if output_file and os.path.exists(output_file) and not overwrite:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_result = json.load(f)

            # 判断是否完整（例如 len(existing_result) == 1 代表任务已完成）
            if isinstance(existing_result, dict) and 'overall_summary' in existing_result:
                print(f"Existing result found at {output_file}. Task already done, skipping API call.")
                if print_report:
                    print_deepseek_summary(existing_result)
                return existing_result
            else:
                print(f"Existing file found but incomplete, continuing to summarize reasoning...")
        except Exception as e:
            print(f"Error reading existing result file: {e}. Will regenerate summary.")

    # ==== [3] 若未完成任务，则调用 DeepSeek API ====
    summary = summarize_reasoning_with_deepseek(samples, api_key, temperature, top_p)

    # ==== [4] 保存结果 ====
    if output_file:
        try:
            if not overwrite and os.path.exists(output_file):
                base_dir = os.path.dirname(output_file)
                base_name = os.path.basename(output_file)
                name, ext = os.path.splitext(base_name)
                safe_name = f"{name}_{attr or 'new'}{ext or '.json'}"
                output_file = os.path.join(base_dir, safe_name)
                print(f"Output exists. Saving to {output_file} instead to avoid overwriting.")

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"Summary saved to {output_file}")
        except Exception as e:
            print(f"Error saving summary to file: {e}")

    # ==== [5] 打印报告 ====
    if print_report:
        print_deepseek_summary(summary)

    return summary



def example_usage():

    api_key = "sk-PxM40luD13UVKLhp6k3zenHC2XPASEi5uazXuXsCfTrQ3hUQ"


    print("=== 思考轨迹分析功能示例 (使用 DeepSeek API) ===\n")
    
    # 示例1: 分析ETTh1数据集的所有样本
    print("1. Analyzing all samples for ETTh1...")
    analyze_memory_reasoning(
        data_name='ETTh1',
        api_key=api_key,
        attr='HUFL',
        look_back=96,
        pred_window=96,
        output_dir='/data/songliv/TS/TimeReasoner/Memory/summary_outputs',
        overwrite=False,
        number=1
    )


if __name__ == "__main__":
    # 运行示例
    example_usage()

