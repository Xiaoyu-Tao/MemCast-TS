import pandas as pd
import numpy as np
import json
import re
from utils.metrics import MSE, MAE


def parse_prediction_from_answer(answer_text):
    """
    从answer文本中解析预测值
    
    Args:
        answer_text: answer字段的文本内容
        
    Returns:
        dates: 日期列表
        values: 预测值列表
    """
    dates = []
    values = []
    
    # 提取```之间的内容
    pattern = r'```(.*?)```'
    matches = re.findall(pattern, answer_text, re.DOTALL)
    
    if matches:
        content = matches[0].strip()
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('<') and not line.startswith('>'):
                # 解析每一行: 日期  值
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        # 日期格式: YYYY-MM-DD HH:MM:SS
                        date_str = parts[0] + ' ' + parts[1]
                        value = float(parts[2])
                        dates.append(date_str)
                        values.append(value)
                    except (ValueError, IndexError):
                        continue
    
    return dates, values


def load_ground_truth(data_name, attr, look_back, pred_window, number):
    """
    加载真实值数据
    
    Args:
        data_name: 数据集名称 (如 'ETTh1')
        attr: 属性名称 (如 'HUFL')
        look_back: 回看窗口大小
        pred_window: 预测窗口大小
        number: 样本序号 (0-9)
        
    Returns:
        dates: 真实值对应的日期
        true_values: 真实值
    """
    data_dir = f'/data/songliv/TS/datasets/Single-mode/{data_name}.csv'
    data = pd.read_csv(data_dir)
    
    # 截取相关数据段
    data = data[12 * 30 * 24 + 4 * 30 * 24 - look_back : 12 * 30 * 24 + 8 * 30 * 24]
    
    # 获取指定样本的真实值
    # 训练数据: number * look_back : (number + 1) * look_back
    # 测试数据(真实值): (number + 1) * look_back : (number + 1) * look_back + pred_window
    start_idx = (number + 1) * look_back
    end_idx = start_idx + pred_window
    
    true_data = data.iloc[start_idx:end_idx]
    dates = true_data['date'].tolist()
    true_values = true_data[attr].tolist()
    
    return dates, true_values


def evaluate_single_result(result_file, data_name, attr, look_back, pred_window):
    """
    评估单个结果文件
    
    Args:
        result_file: 结果JSON文件路径
        data_name: 数据集名称
        attr: 属性名称
        look_back: 回看窗口大小
        pred_window: 预测窗口大小
        
    Returns:
        评估结果字典
    """
    # 读取预测结果
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    evaluation_results = []
    
    for item in results:
        index = item['index']
        answer = item.get('answer', '')
        
        # 解析预测值
        # 确保answer是字符串类型
        if isinstance(answer, dict):
            answer_text = str(answer)
        elif isinstance(answer, list):
            answer_text = str(answer)
        elif answer is None:
            print(f"Warning: answer is None for index {index}")
            answer_text = ""
        else:
            answer_text = str(answer)
            
        try:
            pred_dates, pred_values = parse_prediction_from_answer(answer_text)
        except Exception as e:
            print(f"Error parsing answer for index {index}: {e}")
            print(f"Answer type: {type(answer)}")
            print(f"Answer content: {answer_text[:200]}...")  # 只显示前200个字符
            pred_dates, pred_values = [], []
        original_pred_len = len(pred_values)  # 保存原始预测长度
        
        # 加载真实值
        true_dates, true_values = load_ground_truth(data_name, attr, look_back, pred_window, index)
        
        # 计算长度差异
        length_diff = original_pred_len - len(true_values)
        length_status = "匹配" if length_diff == 0 else ("过长" if length_diff > 0 else "过短")
        
        # 检查长度是否匹配
        if len(pred_values) != len(true_values):
            if length_diff > 0:
                print(f"⚠️  Warning: Index {index} - 预测值过长! 预测长度 {len(pred_values)} > 真实长度 {len(true_values)} (多了 {length_diff} 个)")
            else:
                print(f"⚠️  Warning: Index {index} - 预测值过短! 预测长度 {len(pred_values)} < 真实长度 {len(true_values)} (少了 {-length_diff} 个)")
            # 取最小长度进行评估
            min_len = min(len(pred_values), len(true_values))
            pred_values = pred_values[:min_len]
            true_values = true_values[:min_len]
        
        # 计算指标
        if len(pred_values) > 0 and len(true_values) > 0:
            mse = MSE(true_values, pred_values)
            mae = MAE(true_values, pred_values)
            rmse = np.sqrt(mse)
            
            evaluation_results.append({
                'index': index,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'pred_length': original_pred_len,
                'true_length': len(true_values),
                'length_diff': length_diff,
                'length_status': length_status
            })
        else:
            print(f"⚠️  Warning: Index {index} - no valid predictions or true values")
            
            evaluation_results.append({
                'index': index,
                'mse': None,
                'mae': None,
                'rmse': None,
                'pred_length': original_pred_len,
                'true_length': len(true_values),
                'length_diff': length_diff,
                'length_status': length_status
            })
    
    return evaluation_results


def evaluate_all_results(result_dir, data_name, look_back=96, pred_window=96):
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
    import os
    
    attrs = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    
    all_evaluations = {}
    
    # 按属性分组，收集所有片段的结果
    for attr in attrs:
        print(f"\n{'='*60}")
        print(f"Evaluating {attr} - Collecting all segment results...")
        print(f"{'='*60}")
        
        attr_all_results = []  # 收集该属性的所有结果
        
        # 遍历所有片段
        for i in range(88):
            result_file = os.path.join(result_dir, f'result_{attr}_{data_name}_{look_back}_{pred_window}_{i}.json')
            
            if os.path.exists(result_file):
                print(f"  Processing segment {i}...")
                eval_results = evaluate_single_result(result_file, data_name, attr, look_back, pred_window)
                attr_all_results.extend(eval_results)
            else:
                print(f"  Segment {i} not found, skipping...")
        
        # 计算该属性的汇总指标
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
            print(f"  Available segments: {len(attr_all_results)}/118")
            
            # 打印每个样本的详细结果
            print(f"\n  Individual samples:")
            for r in attr_all_results:
                if r['mse'] is not None:
                    length_info = f"[{r['length_status']}]" if r['length_status'] != "匹配" else ""
                    print(f"    Index {r['index']}: MSE={r['mse']:.6f}, MAE={r['mae']:.6f}, RMSE={r['rmse']:.6f} {length_info}")
                    if r['length_status'] != "匹配":
                        print(f"               长度: 预测={r['pred_length']}, 真实={r['true_length']}, 差异={r['length_diff']:+d}")
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
    
    # 计算总体平均
    valid_attrs = [k for k, v in all_evaluations.items() if v is not None]
    if valid_attrs:
        overall_mse = np.mean([all_evaluations[k]['average_mse'] for k in valid_attrs])
        overall_mae = np.mean([all_evaluations[k]['average_mae'] for k in valid_attrs])
        overall_rmse = np.mean([all_evaluations[k]['average_rmse'] for k in valid_attrs])
        
        print(f"{'Overall':<10} {overall_mse:<15.6f} {overall_mae:<15.6f} {overall_rmse:<15.6f}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # 评估ETTh1数据集的结果
    data_name = 'ETTh1'
    # for i in [7]:
    #     result_dir = f'/data/songliv/TS/TimeReasoner/output_{i}/result/{data_name}'
    #     output_file = f'/data/songliv/TS/TimeReasoner/output_{i}/evaluation_results.json'
    result_dir = f'/data/songliv/TS/TimeReasoner/output_3_summery/result/{data_name}'
    output_file = f'/data/songliv/TS/TimeReasoner/output_3_summery/evaluation_results.json'
    print(result_dir)
    look_back = 96
    pred_window = 96
    
    all_evaluations = evaluate_all_results(result_dir, data_name, look_back, pred_window)
    print_summary(all_evaluations)
    
    # 保存评估结果
    
    with open(output_file, 'w') as f:
        # 转换numpy类型为python原生类型以便JSON序列化
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
                            'length_status': r['length_status']
                        }
                        for r in results['individual_results']
                    ]
                }
            else:
                serializable_results[attr] = None
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"✅ Evaluation results saved to: {output_file}")

