import os
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.metrics import MSE, MAE
from utils.tools import visual

def parse_prediction_from_answer(answer_text):
    """
    从 LLM 的 answer 文本中解析预测结果，兼容空格分隔或逗号分隔格式
    """
    dates, values = [], []
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, answer_text, re.DOTALL | re.IGNORECASE)

    if not match:
        return dates, values

    content = match.group(1).strip()

    if content.startswith("```"):
        content = re.sub(r'^```[^\n]*\n', '', content, count=1)
        content = re.sub(r'\n?```$', '', content, count=1)
        content = content.strip()

    lines = content.split('\n')

    for line in lines:
        line = line.strip()
        if not line or line.startswith(('date', '<', '>')):  # 跳过表头与标签
            continue

        # 尝试解析逗号分隔
        if ',' in line:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                date_str, val_str = parts[0], parts[1]
            else:
                continue
        else:
            # 尝试空格分隔格式
            parts = line.split()
            if len(parts) >= 3:
                date_str = parts[0] + ' ' + parts[1]
                val_str = parts[2]
            elif len(parts) == 2:
                date_str, val_str = parts
            else:
                continue

        try:
            value = float(val_str)
            dates.append(date_str)
            values.append(value)
        except ValueError:
            continue

    return dates, values



def load_ground_truth(data_name, attr, look_back, pred_window, number):
    """
    加载指定样本的历史输入与真实预测数据
    
    Args:
        data_name: 数据集名称 (如 'ETTh1')
        attr: 属性名称 (如 'HUFL')
        look_back: 回看窗口大小
        pred_window: 预测窗口大小
        number: 样本序号 (0-9)
        
    Returns:
        history_dates: 输入序列对应的日期
        history_values: 输入序列的数值
        future_dates: 真实值对应的日期
        future_values: 真实值
    """
    data_dir = f'/data/songliv/TS/datasets/EPF/{data_name}.csv'
    data = pd.read_csv(data_dir)
    
    # 截取相关数据段
    # end_idx = int(len(data) * 0.8) + look_back
    # if end_idx > len(data):
    #     end_idx = len(data)
    # data = data[:end_idx]
    start_idx = int(len(data) * 0.8) - look_back
    if start_idx < 0:
        start_idx = 0
    data = data[start_idx:]
    # 获取指定样本的真实值
    # 滑动窗口设置为24，每次滑动24个时间步
    # 窗口0: 数据索引 [0:look_back]，真实值应该在 [look_back:look_back+pred_window]
    # 窗口1: 数据索引 [24:24+look_back]，真实值应该在 [24+look_back:24+look_back+pred_window]
    # 窗口n: 数据索引 [n*24:n*24+look_back]，真实值应该在 [n*24+look_back:n*24+look_back+pred_window]
    slide_window = 24
    start_idx = number * slide_window + look_back
    end_idx = start_idx + pred_window
    
    # 目标列可能存在大小写或空格差异，尝试匹配
    target_col = attr
    if target_col not in data.columns:
        candidates = [
            col for col in data.columns
            if attr.lower().strip() == col.lower().strip()
            or attr.lower().strip() in col.lower().strip()
            or col.lower().strip() in attr.lower().strip()
        ]
        if candidates:
            target_col = candidates[0]
    
    history_start_idx = number * slide_window
    history_end_idx = history_start_idx + look_back
    if history_start_idx < 0:
        history_start_idx = 0
    if history_end_idx > len(data):
        history_end_idx = len(data)
        history_start_idx = max(0, history_end_idx - look_back)
    
    history_data = data.iloc[history_start_idx:history_end_idx]
    future_data = data.iloc[start_idx:end_idx]
    
    history_dates = history_data['date'].tolist()
    history_values = history_data[target_col].tolist() if target_col in history_data else []
    
    future_dates = future_data['date'].tolist()
    future_values = future_data[target_col].tolist() if target_col in future_data else []
    
    return history_dates, history_values, future_dates, future_values


def evaluate_single_result(result_file, data_name, attr, look_back, pred_window,i):
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

    result_dir = os.path.dirname(result_file)
    output_root = os.path.abspath(os.path.join(result_dir, os.pardir, os.pardir))
    figures_root = os.path.join(output_root, 'figures', data_name, attr)
    os.makedirs(figures_root, exist_ok=True)
    segment_id = f'{i}'
    
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
        
        # 加载输入及真实值
        history_dates, history_values, true_dates, true_values = load_ground_truth(
            data_name,
            attr,
            look_back,
            pred_window,
            i
        )
        
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
            pred_dates = pred_dates[:min_len]
            true_dates = true_dates[:min_len]
        
        # 计算指标
        if len(pred_values) > 0 and len(true_values) > 0:
            future_numeric = []
            for value in true_values:
                try:
                    future_numeric.append(float(value))
                except (TypeError, ValueError):
                    future_numeric.append(np.nan)

            pred_numeric = []
            for value in pred_values:
                try:
                    pred_numeric.append(float(value))
                except (TypeError, ValueError):
                    pred_numeric.append(np.nan)

            future_array = np.array(future_numeric, dtype=float)
            pred_array = np.array(pred_numeric, dtype=float)
            valid_mask = np.isfinite(future_array) & np.isfinite(pred_array)

            if not np.any(valid_mask):
                print(f"⚠️  Warning: Index {index} - no finite values for metric computation")
                evaluation_results.append({
                    'index': index,
                    'mse': None,
                    'mae': None,
                    'rmse': None,
                    'pred_length': original_pred_len,
                    'true_length': int(len(future_numeric)),
                    'length_diff': length_diff,
                    'length_status': length_status,
                    'figure_path': None,
                    'figure_pdf_path': None
                })
                continue

            future_clean = future_array[valid_mask]
            pred_clean = pred_array[valid_mask]

            mse = MSE(future_clean, pred_clean)
            mae = MAE(future_clean, pred_clean)
            rmse = np.sqrt(mse)

            hist_numeric = []
            for value in history_values:
                try:
                    hist_numeric.append(float(value))
                except (TypeError, ValueError):
                    hist_numeric.append(np.nan)

            hist_array = np.array(hist_numeric, dtype=float)
            future_plot = np.array(future_numeric, dtype=float)
            pred_plot = np.array(pred_numeric, dtype=float)

            fig, ax = plt.subplots(figsize=(10, 4.8))

            hist_len = len(hist_array)
            fut_len = len(future_plot)

            x_hist = np.arange(-hist_len, 0, 1)
            x_future = np.arange(0, fut_len, 1)

            if hist_len > 0:
                ax.plot(
                    x_hist,
                    hist_array,
                    color='#2ca02c',
                    linewidth=1.4,
                    label='History Input'
                )

            ax.plot(
                x_future,
                future_plot,
                label='GroundTruth',
                color='#1f77b4',
                linewidth=1.4
            )
            ax.plot(
                x_future,
                pred_plot,
                label='Prediction',
                color='#ff7f0e',
                linewidth=1.8
            )

            forecast_marker = -0.5 if hist_len > 0 else -0.1
            ax.axvline(forecast_marker, color='#888', linestyle='--', linewidth=1.1)
            ax.text(
                forecast_marker,
                ax.get_ylim()[1],
                ' Forecast Start',
                fontsize=9,
                color='#555',
                va='top',
                ha='left'
            )

            ax.set_title(
                f'{attr} History & Forecast · {segment_id} · Attempt {index}',
                fontsize=12
            )
            ax.set_xlabel('Time Index')
            ax.set_ylabel(attr)

            tick_positions = []
            tick_labels = []

            if hist_len > 0 and history_dates:
                hist_step = max(1, hist_len // 8)
                for i_tick in range(0, hist_len, hist_step):
                    tick_positions.append(x_hist[i_tick])
                    tick_labels.append(history_dates[i_tick])

            if fut_len > 0 and pred_dates:
                fut_step = max(1, fut_len // 8)
                for i_tick in range(0, fut_len, fut_step):
                    tick_positions.append(x_future[i_tick])
                    tick_labels.append(pred_dates[i_tick])

            if tick_positions:
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(
                    tick_labels,
                    rotation=40,
                    ha='right',
                    fontsize=7
                )

            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)

            plt.tight_layout()

            figure_path = os.path.join(
                figures_root,
                f'{segment_id}_attempt_{index:02d}.png'
            )
            fig.savefig(figure_path, dpi=160, bbox_inches='tight')
            plt.close(fig)

            full_true = np.concatenate([hist_array, future_plot])
            full_pred = np.concatenate([hist_array, pred_plot])

            pdf_path = os.path.join(figures_root, f'{segment_id}_{index}.pdf')
            try:
                visual(full_true.tolist(), full_pred.tolist(), pdf_path)
            except Exception as exc:
                print(f"Warning: could not generate visual PDF for segment {segment_id}, attempt {index}: {exc}")
                pdf_path = None

            figure_rel_path = os.path.relpath(figure_path, start=output_root)
            pdf_rel_path = (
                os.path.relpath(pdf_path, start=output_root)
                if pdf_path is not None else None
            )

            evaluation_results.append({
                'index': index,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'pred_length': original_pred_len,
                'true_length': len(true_values),
                'length_diff': length_diff,
                'length_status': length_status,
                'figure_path': figure_rel_path,
                'figure_pdf_path': pdf_rel_path
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
    attrs = ['OT']
    
    all_evaluations = {}
    
    # 按属性分组，收集所有片段的结果
    for attr in attrs:
        print(f"\n{'='*60}")
        print(f"Evaluating {attr} - Collecting all segment results...")
        print(f"{'='*60}")
        
        attr_all_results = []  # 收集该属性的所有结果
        
        # 遍历所有片段
        for i in range(30):
            result_file = os.path.join(result_dir, f'result_{attr}_{data_name}_{look_back}_{pred_window}_{i}.json')
            
            if os.path.exists(result_file):
                print(f"  Processing segment {i}...")
                eval_results = evaluate_single_result(result_file, data_name, attr, look_back, pred_window,i)
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
    data_name = 'NP'
    # for i in [7]:
    #     result_dir = f'/data/songliv/TS/TimeReasoner/output_{i}/result/{data_name}'
    #     output_file = f'/data/songliv/TS/TimeReasoner/output_{i}/evaluation_results.json'
    result_dir = f'/data/songliv/TS/TimeReasoner/output_see/result/{data_name}'
    output_file = f'/data/songliv/TS/TimeReasoner/output_see/evaluation_results.json'
    print(result_dir)
    look_back = 168
    pred_window = 24
    
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
                            'length_status': r['length_status'],
                            'figure_path': r.get('figure_path'),
                            'figure_pdf_path': r.get('figure_pdf_path')
                        }
                        for r in results['individual_results']
                    ]
                }
            else:
                serializable_results[attr] = None
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"✅ Evaluation results saved to: {output_file}")

