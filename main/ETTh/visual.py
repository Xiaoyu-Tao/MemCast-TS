import pandas as pd
import json
import os
import matplotlib.pyplot as plt
meaning_dict={
               'OT':'Oil Temperature'}


def visual(data_name,attr,look_back,pred_window,number):

    data_dir = '/data/songliv/TS/datasets/Single-mode/' + data_name + '.csv'
    data = pd.read_csv(data_dir)
    # 取前12个月的数据 (假设每小时一个数据点，12个月 = 12 * 30 * 24)
    data = data[12 * 30 * 24 + 4 * 30 * 24 - look_back : 12 * 30 * 24 + 8 * 30 * 24]
    date = data.loc[:, 'date'].to_numpy()
    attr_data = data.loc[:, attr].to_numpy()
    data = pd.DataFrame(date, columns=['date'])
    data[attr] = attr_data
    data_lookback = []
    
    for i in range(30):
        data_lookback.append(data.iloc[i * look_back:(i + 1) * look_back])  # 使用 iloc 进行索引
    
    # 计算整体区间统计信息
    overall_series = data[attr]
    overall_stats = {
        'count': int(overall_series.count()),
        'mean': float(overall_series.mean()),
        'std': float(overall_series.std()),
        'var': float(overall_series.var()),
        'min': float(overall_series.min()),
        'max': float(overall_series.max())
    }
    print(f"[Overall] {attr} stats -> count: {overall_stats['count']}, mean: {overall_stats['mean']:.4f}, std: {overall_stats['std']:.4f}, var: {overall_stats['var']:.4f}, min: {overall_stats['min']:.4f}, max: {overall_stats['max']:.4f}")

    # 计算指定窗口段的统计信息
    if 0 <= number < len(data_lookback):
        seg_series = data_lookback[number][attr]
        seg_stats = {
            'count': int(seg_series.count()),
            'mean': float(seg_series.mean()),
            'std': float(seg_series.std()),
            'var': float(seg_series.var()),
            'min': float(seg_series.min()),
            'max': float(seg_series.max())
        }
        print(f"[Segment {number}] {attr} stats -> count: {seg_stats['count']}, mean: {seg_stats['mean']:.4f}, std: {seg_stats['std']:.4f}, var: {seg_stats['var']:.4f}, min: {seg_stats['min']:.4f}, max: {seg_stats['max']:.4f}")
    else:
        print(f"Segment index out of range: {number}")

    plt.plot(data_lookback[number][attr])
    plt.show()

if __name__ == "__main__":
    visual(data_name='ETTh1',attr='OT',look_back=96,pred_window=96,number=0)
    