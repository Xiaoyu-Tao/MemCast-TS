from utils.api_ouput import deepseek_api_output
import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from prompt.success_cases_extraction import time_series_memory_extraction_prompt
from prompt.fail_cases_extraction import time_series_failure_extraction_prompt
from prompt.cases_refine_prompt import time_series_memory_extraction_prompt as cases_refine_prompt
import hashlib
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    HashingVectorizer = None
    StandardScaler = None
    cosine_similarity = None

def MAE(predicted, actual) -> float:
    predicted_arr = np.asarray(predicted, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    return float(np.mean(np.abs(predicted_arr - actual_arr)))

def MSE(predicted, actual) -> float:
    predicted_arr = np.asarray(predicted, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    return float(np.mean((predicted_arr - actual_arr) ** 2))

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

def retrieve_similar_examples(current_data, historical_examples, top_k=5, max_index: Optional[int] = None):
    """
    基于时间序列特征相似性检索相似的样例
    """
    if not historical_examples:
        return []
    if max_index is not None:
        try:
            upper = int(max_index)
            historical_examples = [e for e in historical_examples if isinstance(e, dict) and isinstance(e.get("index"), int) and int(e.get("index")) <= upper]
        except Exception:
            pass
        if not historical_examples:
            return []
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
    scaler.fit(historical_features)
    current_features_scaled = scaler.transform(current_features.reshape(1, -1))
    historical_features_scaled = scaler.transform(historical_features)
    
    if debug:
        print(f"[DEBUG] current_features: {current_features}")
        print(f"[DEBUG] historical_features shape: {historical_features.shape}")
        print(f"[DEBUG] historical_features sample: {historical_features[:3]}")
        print(f"[DEBUG] scaler.mean_: {scaler.mean_}")
        print(f"[DEBUG] scaler.scale_: {scaler.scale_}")
        print(f"[DEBUG] any NaN in current_features_scaled: {np.isnan(current_features_scaled).any()}")
        print(f"[DEBUG] any NaN in historical_features_scaled: {np.isnan(historical_features_scaled).any()}")
        print(f"[DEBUG] current_features_scaled: {current_features_scaled}")
    
    # 计算余弦相似度
    similarities = cosine_similarity(current_features_scaled, historical_features_scaled)[0]
    if debug:
        print(f"[DEBUG] similarities: {similarities}")
    
    # 获取最相似的top_k个样例
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    similar_examples = []
    for idx in top_indices:
        similar_examples.append({
            'example': historical_examples[idx],
            'similarity': similarities[idx],
            'index': historical_examples[idx].get('index', idx)
        })
    
    return similar_examples

def _extract_forecast_lines(answer_text):
    """
    Extract forecast lines from the model answer.

    Returns:
        tuple[list[str] | None, str]: (lines, error_message)
    """
    if not answer_text or not answer_text.strip():
        return None, "Answer text is empty."

    # 优先匹配标准格式
    match = re.search(r"<answer>\s*```(.*?)```\s*</answer>", answer_text, re.DOTALL | re.IGNORECASE)

    # 若未匹配成功，再尝试宽松格式
    if not match:
        match = re.search(r"<answer>(.*?)</answer>", answer_text, re.DOTALL | re.IGNORECASE)

    if not match:
        return None, "Answer does not contain a recognizable <answer> ... </answer> block."


    # match = re.search(r"<answer>\s*```(.*?)```\s*</answer>", answer_text, re.DOTALL | re.IGNORECASE)
    # if not match:
    #     return None, "Answer does not follow the required <answer> ```...``` </answer> format."

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

def evaluate_prediction_quality(predicted, actual):
    """
    评估预测质量，返回质量分数
    """
    if len(predicted) == 0 or len(actual) == 0:
        return None, None, None
    
    # 确保长度一致
    min_len = min(len(predicted), len(actual))
    predicted = predicted[:min_len]
    actual = actual[:min_len]
    
    mae = MAE(predicted, actual)
    mse = MSE(predicted, actual)
    rmse = float(np.sqrt(mse))

    return mse, mae, rmse

def _load_dataset(data_name: str, dataset_dir: str = "datasets", train_ratio: float = 0.8) -> pd.DataFrame:
    dataset_path = os.path.join(dataset_dir, f"{data_name}.csv")
    data = pd.read_csv(dataset_path)
    data.columns = data.columns.str.strip()

    end_idx = int(len(data) * train_ratio)
    end_idx = min(max(end_idx, 1), len(data))
    return data.iloc[:end_idx].reset_index(drop=True)

def _get_ground_truth(
    data: pd.DataFrame,
    attr: str,
    look_back: int,
    pred_window: int,
    number: int,
    slide_window: int = 24,
) -> Optional[List[float]]:
    if attr not in data.columns:
        return None
    start_idx = number * slide_window
    pred_start = start_idx + look_back
    pred_end = pred_start + pred_window
    if pred_start < 0 or pred_end > len(data):
        return None
    series = data[attr].iloc[pred_start:pred_end]
    if len(series) != pred_window:
        return None
    if series.isna().any():
        return None
    return series.astype(float).tolist()

def _get_lookback_window(
    data: pd.DataFrame,
    attr: str,
    look_back: int,
    number: int,
    slide_window: int = 24,
) -> Optional[List[float]]:
    if attr not in data.columns:
        return None
    start_idx = number * slide_window
    hist_start = start_idx
    hist_end = hist_start + look_back
    if hist_start < 0 or hist_end > len(data):
        return None
    series = data[attr].iloc[hist_start:hist_end]
    if len(series) != look_back:
        return None
    if series.isna().any():
        return None
    return series.astype(float).tolist()

def _extract_feature_vector(
    data_name: str,
    attr: str,
    look_back: int,
    number: int,
    dataset_dir: str = "datasets",
    train_ratio: float = 0.8,
    slide_window: int = 24,
) -> Optional[np.ndarray]:
    try:
        data = _load_dataset(data_name=data_name, dataset_dir=dataset_dir, train_ratio=train_ratio)
        series = _get_lookback_window(
            data=data,
            attr=attr,
            look_back=int(look_back),
            number=int(number),
            slide_window=int(slide_window),
        )
        if series is None:
            return None
        feat = extract_time_series_features(np.asarray(series, dtype=float))
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        if feat.size == 0 or np.isnan(feat).any():
            return None
        # feat = _normalize_vector(feat)
        # if float(np.linalg.norm(feat)) <= 0:
        #     return None
        return feat
    except Exception:
        return None

def _get_ground_truth_lines(
    data: pd.DataFrame,
    attr: str,
    look_back: int,
    pred_window: int,
    number: int,
    slide_window: int = 24,
) -> Optional[List[str]]:
    if "date" not in data.columns or attr not in data.columns:
        return None
    start_idx = number * slide_window
    pred_start = start_idx + look_back
    pred_end = pred_start + pred_window
    if pred_start < 0 or pred_end > len(data):
        return None
    window = data.iloc[pred_start:pred_end][["date", attr]]
    if len(window) != pred_window:
        return None
    if window.isna().any().any():
        return None
    lines: List[str] = []
    for _, row in window.iterrows():
        lines.append(f"{row['date']},{float(row[attr])}")
    return lines

def _get_covariate_window_text(
    data: pd.DataFrame,
    covariates: List[str],
    look_back: int,
    pred_window: int,
    number: int,
    slide_window: int = 24,
) -> str:
    if "date" not in data.columns:
        return ""
    cov_cols = [c for c in covariates if c in data.columns]
    if not cov_cols:
        return ""

    start_idx = number * slide_window
    hist_start = start_idx
    hist_end = hist_start + look_back
    fut_start = hist_end
    fut_end = fut_start + pred_window
    if hist_start < 0 or fut_end > len(data):
        return ""

    header = "date," + ",".join(cov_cols)

    def fmt_window(df: pd.DataFrame) -> str:
        lines = [header]
        for _, row in df.iterrows():
            vals: List[str] = []
            for c in cov_cols:
                v = row[c]
                if pd.isna(v):
                    vals.append("NA")
                else:
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        vals.append(str(v))
                    else:
                        vals.append(str(v))
            lines.append(f"{row['date']}," + ",".join(vals))
        return "\n".join(lines)

    hist_df = data.iloc[hist_start:hist_end][["date"] + cov_cols]
    fut_df = data.iloc[fut_start:fut_end][["date"] + cov_cols]

    return (
        "# Exogenous Covariate Values (Grid load forecast / Wind power forecast)\n"
        "## History Window\n"
        f"{fmt_window(hist_df)}\n"
        "## Future Window\n"
        f"{fmt_window(fut_df)}"
    )

def _clean_json_text(text: str) -> str:
    if not text:
        return ""
    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.replace("```", "").strip()
    return text.strip()

def _lift_data_pattern_from_distilled(parsed: Any) -> Tuple[Optional[str], Any]:
    data_pattern: Optional[str] = None
    distilled: Any = parsed
    if isinstance(parsed, dict):
        dp = parsed.get("data_pattern")
        if isinstance(dp, str) and dp.strip():
            data_pattern = dp.strip()
        if "distilled" in parsed:
            distilled = parsed.get("distilled")

    if isinstance(distilled, list):
        cleaned: List[Any] = []
        for item in distilled:
            if isinstance(item, dict):
                dp = item.get("data_pattern")
                if data_pattern is None and isinstance(dp, str) and dp.strip():
                    data_pattern = dp.strip()
                if "data_pattern" in item:
                    item = {k: v for k, v in item.items() if k != "data_pattern"}
            cleaned.append(item)
        distilled = cleaned
    elif isinstance(distilled, dict):
        dp = distilled.get("data_pattern")
        if data_pattern is None and isinstance(dp, str) and dp.strip():
            data_pattern = dp.strip()
        if "data_pattern" in distilled:
            distilled = {k: v for k, v in distilled.items() if k != "data_pattern"}

    return data_pattern, distilled

def _save_data_pattern(
    data_name: str,
    attr: str,
    look_back: int,
    pred_window: int,
    index: int,
    data_pattern: str,
    case_type: str,
) -> None:
    pattern_dir = os.path.join("Memory", "pattern", "EPF", data_name)
    os.makedirs(pattern_dir, exist_ok=True)
    pattern_path = os.path.join(
        pattern_dir,
        f"pattern_{attr}_{data_name}_{look_back}_{pred_window}_{index}.json",
    )
    payload = {
        "data_name": data_name,
        "attr": attr,
        "look_back": look_back,
        "pred_window": pred_window,
        "index": index,
        "case_type": case_type,
        "data_pattern": data_pattern,
    }
    try:
        with open(pattern_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Failed to save data_pattern to {pattern_path}: {e}")


def _stable_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= 0:
        return v
    return v / n

def _distilled_to_text(summary: Dict[str, Any]) -> str:
    chunks: List[str] = []

    refined = summary.get("refined_distilled")
    if isinstance(refined, list) and refined:
        for item in refined:
            if not isinstance(item, dict):
                continue
            if "experience" in item or "success_insight" in item or "meta_logic" in item:
                parts = [
                    str(item.get("experience") or ""),
                    str(item.get("success_insight") or ""),
                    json.dumps(item.get("meta_logic") or {}, ensure_ascii=False),
                ]
                txt = "\n".join([p for p in parts if p]).strip()
                if txt:
                    chunks.append(txt)
            elif "failure_analysis" in item or "preventative_rule" in item or "meta_logic" in item:
                parts = [
                    str(item.get("failure_analysis") or ""),
                    str(item.get("preventative_rule") or ""),
                    json.dumps(item.get("meta_logic") or {}, ensure_ascii=False),
                ]
                txt = "\n".join([p for p in parts if p]).strip()
                if txt:
                    chunks.append(txt)
            else:
                chunks.append(json.dumps(item, ensure_ascii=False))
        chunks = [c for c in chunks if c and c.strip()]
        return "\n\n---\n\n".join(chunks).strip()

    def add_case_entries(entries: Any):
        if not isinstance(entries, list):
            return
        for e in entries:
            if not isinstance(e, dict):
                continue
            distilled = e.get("distilled")
            if isinstance(distilled, list):
                for item in distilled:
                    if not isinstance(item, dict):
                        continue
                    if "experience" in item or "success_insight" in item or "meta_logic" in item:
                        parts = [
                            str(item.get("experience") or ""),
                            str(item.get("success_insight") or ""),
                            json.dumps(item.get("meta_logic") or {}, ensure_ascii=False),
                        ]
                        chunks.append("\n".join([p for p in parts if p]).strip())
                    elif "failure_analysis" in item or "preventative_rule" in item or "meta_logic" in item:
                        parts = [
                            str(item.get("failure_analysis") or ""),
                            str(item.get("preventative_rule") or ""),
                            json.dumps(item.get("meta_logic") or {}, ensure_ascii=False),
                        ]
                        chunks.append("\n".join([p for p in parts if p]).strip())
                    else:
                        chunks.append(json.dumps(item, ensure_ascii=False))
            elif isinstance(distilled, dict):
                chunks.append(json.dumps(distilled, ensure_ascii=False))

    add_case_entries(summary.get("good_cases_distilled"))
    add_case_entries(summary.get("bad_cases_distilled"))

    chunks = [c for c in chunks if c and c.strip()]
    return "\n\n---\n\n".join(chunks).strip()

def _embed_text(
    text: str,
    api_key: str,
    base_url: Optional[str] = None,
    model: str = "text-embedding-3-small",
    dim: int = 1024,
) -> Optional[np.ndarray]:
    t = (text or "").strip()
    if not t:
        return None

    if OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            resp = client.embeddings.create(model=model, input=t)
            vec = resp.data[0].embedding
            return _normalize_vector(np.asarray(vec, dtype=np.float32))
        except Exception:
            pass

    if HashingVectorizer is None:
        return None

    try:
        hv = HashingVectorizer(
            n_features=int(dim),
            alternate_sign=False,
            norm=None,
            analyzer="char",
            ngram_range=(3, 5),
        )
        x = hv.transform([t])
        v = x.toarray().astype(np.float32)[0]
        v = _normalize_vector(v)
        if float(np.linalg.norm(v)) <= 0:
            return None
        return v
    except Exception:
        return None

def _load_vector_db(db_path: str) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    meta_path = os.path.splitext(db_path)[0] + ".json"
    if not os.path.exists(db_path) or not os.path.exists(meta_path):
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), []
    try:
        npz = np.load(db_path, allow_pickle=False)
        if "cot_vectors" in npz:
            cot_vectors = np.asarray(npz["cot_vectors"], dtype=np.float32)
        elif "vectors" in npz:
            cot_vectors = np.asarray(npz["vectors"], dtype=np.float32)
        else:
            cot_vectors = np.zeros((0, 0), dtype=np.float32)

        feat_vectors = np.asarray(npz["feat_vectors"], dtype=np.float32) if "feat_vectors" in npz else None
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        entries = meta.get("entries", []) if isinstance(meta, dict) else []
        if not isinstance(entries, list):
            entries = []
        if cot_vectors.ndim != 2:
            cot_vectors = np.zeros((0, 0), dtype=np.float32)
        if feat_vectors is None:
            feat_vectors = np.zeros((cot_vectors.shape[0], 0), dtype=np.float32) if cot_vectors.size else np.zeros((0, 0), dtype=np.float32)
        elif feat_vectors.ndim != 2:
            feat_vectors = np.zeros((cot_vectors.shape[0], 0), dtype=np.float32) if cot_vectors.size else np.zeros((0, 0), dtype=np.float32)

        n = cot_vectors.shape[0]
        if feat_vectors.shape[0] != n:
            n = min(n, feat_vectors.shape[0])
        if len(entries) != n:
            n = min(n, len(entries))
        cot_vectors = cot_vectors[:n]
        feat_vectors = feat_vectors[:n]
        entries = entries[:n]
        return cot_vectors, feat_vectors, entries
    except Exception:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), []

def _save_vector_db(db_path: str, cot_vectors: np.ndarray, feat_vectors: np.ndarray, entries: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    meta_path = os.path.splitext(db_path)[0] + ".json"
    cot_vectors = np.asarray(cot_vectors, dtype=np.float32)
    feat_vectors = np.asarray(feat_vectors, dtype=np.float32)
    np.savez_compressed(db_path, cot_vectors=cot_vectors, feat_vectors=feat_vectors, vectors=cot_vectors)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, ensure_ascii=False, indent=2)

def _increase_entry_confidence(meta_path: str, target_ids: List[str], delta: float = 0.01) -> int:
    if not meta_path or not os.path.exists(meta_path):
        print(f"[CONF] skip meta_missing meta={meta_path}")
        return 0
    if not target_ids:
        print(f"[CONF] skip empty_target_ids meta={meta_path}")
        return 0
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        entries = meta.get("entries", []) if isinstance(meta, dict) else []
        if not isinstance(entries, list):
            print(f"[CONF] skip invalid_entries meta={meta_path} type={type(entries)}")
            return 0
        targets = {str(x) for x in target_ids if x is not None}
        if not targets:
            print(f"[CONF] skip empty_targets meta={meta_path}")
            return 0
        updated = 0
        updates: List[Tuple[str, float, float]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_id = entry.get("id")
            entry_id_str = str(entry_id) if entry_id is not None else ""
            if entry_id_str in targets:
                try:
                    current_conf = float(entry.get("confidence", 0))
                except Exception:
                    current_conf = 0.0
                new_conf = current_conf + float(delta)
                entry["confidence"] = new_conf
                updated += 1
                updates.append((entry_id_str, current_conf, new_conf))
        if updated > 0:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"entries": entries}, f, ensure_ascii=False, indent=2)
            print(f"[CONF] updated={updated} delta={delta} meta={meta_path}")
            for entry_id_str, old_conf, new_conf in updates:
                print(f"[CONF] id={entry_id_str} {old_conf:.6f} -> {new_conf:.6f}")
        else:
            sample_targets = sorted(list(targets))[:5]
            print(f"[CONF] updated=0 delta={delta} targets={len(targets)} sample_targets={sample_targets} meta={meta_path}")
        return updated
    except Exception:
        print(f"[CONF] error meta={meta_path}")
        return 0

def _is_good_case_summary(summary: Dict[str, Any], number: Optional[int]) -> bool:
    if number is None or not isinstance(summary, dict):
        return False
    good_cases = summary.get("good_cases_distilled")
    if isinstance(good_cases, list):
        for item in good_cases:
            if isinstance(item, dict) and item.get("file_index") == number:
                return True
    return False

def _append_jsonl(log_path: str, payload: Dict[str, Any]):
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return

def _print_dedup_event(payload: Dict[str, Any]):
    if not isinstance(payload, dict):
        return
    action = payload.get("action")
    sample_id = payload.get("sample_id")
    db_path = payload.get("db_path")
    top_sim = payload.get("top1_similarity")
    top_id = payload.get("top1_id")
    top_cot = payload.get("top1_cot_similarity")
    top_feat = payload.get("top1_feat_similarity")
    removed_count = payload.get("removed_count")
    removed_ids = payload.get("removed_ids")
    if action == "replace":
        print(
            f"[DEDUP] replace sample_id={sample_id} top1_sim={top_sim} cot_sim={top_cot} feat_sim={top_feat} top1_id={top_id} "
            f"removed_count={removed_count} removed_ids={removed_ids} db={db_path}"
        )
    elif action == "refine_add":
        refined_from = payload.get("refined_from")
        refined_count = payload.get("refined_count")
        print(
            f"[DEDUP] refine_add sample_id={sample_id} top1_sim={top_sim} cot_sim={top_cot} feat_sim={top_feat} top1_id={top_id} "
            f"refined_from={refined_from} refined_count={refined_count} db={db_path}"
        )
    elif action == "add":
        print(f"[DEDUP] add sample_id={sample_id} top1_sim={top_sim} cot_sim={top_cot} feat_sim={top_feat} top1_id={top_id} db={db_path}")

def _load_summary_file(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _default_refined_summary_path(
    refine_dir: str,
    data_name: str,
    sample_id: str,
    original_summary_file: Optional[str],
) -> str:
    base = os.path.basename(original_summary_file) if original_summary_file else ""
    if base and base.lower().endswith(".json"):
        file_name = base
    else:
        safe = re.sub(r"[^0-9a-zA-Z._-]+", "_", sample_id or "sample")
        file_name = f"refined_{safe}.json"
    return os.path.join(refine_dir, data_name, file_name)

def _save_refined_summary(path: str, summary_obj: Dict[str, Any]) -> bool:
    if not path:
        return False
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary_obj, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False

def _refine_distilled_with_deepseek(
    api_key: str,
    combined_distilled_text: str,
    temperature: float = 0.2,
    top_p: float = 1.0,
) -> Optional[List[Dict[str, Any]]]:
    text = (combined_distilled_text or "").strip()
    if not text:
        return None
    try:
        model = deepseek_api_output(api_key=api_key, temperature=temperature, top_p=top_p)
        prompt = cases_refine_prompt.format(text=text)
        _, out = model(prompt)
        cleaned = _clean_json_text(out)
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            items = [x for x in obj if isinstance(x, dict)]
            return items if items else None
        if isinstance(obj, dict):
            lst = obj.get("refined_distilled")
            if isinstance(lst, list):
                items = [x for x in lst if isinstance(x, dict)]
                return items if items else None
        return None
    except Exception:
        return None

def _extract_case_level_data_pattern(summary: Dict[str, Any], number: Optional[int]) -> Optional[str]:
    if not isinstance(summary, dict):
        return None
    for key in ("good_cases_distilled", "bad_cases_distilled"):
        items = summary.get(key) or []
        if not isinstance(items, list) or not items:
            continue
        candidates: List[Any] = []
        if number is not None:
            candidates = [x for x in items if isinstance(x, dict) and x.get("file_index") == number]
        if not candidates:
            candidates = [items[0]]
        for entry in candidates:
            if not isinstance(entry, dict):
                continue
            dp = entry.get("data_pattern")
            if isinstance(dp, str) and dp.strip():
                return dp.strip()
            distilled = entry.get("distilled")
            if isinstance(distilled, list):
                for d in distilled:
                    if not isinstance(d, dict):
                        continue
                    dp = d.get("data_pattern")
                    if isinstance(dp, str) and dp.strip():
                        return dp.strip()
    return None

def _dedup_check_and_update_db(
    summary: Dict[str, Any],
    api_key: str,
    db_path: str,
    sample_id: str,
    cot_replace_threshold: float = 0.9,
    feat_replace_threshold: float = 0.9,
    cot_refine_threshold: float = 0.8,
    feat_refine_threshold: float = 0.8,
    embedding_base_url: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    feature_vector: Optional[np.ndarray] = None,
    log_path: Optional[str] = None,
    summary_file: Optional[str] = None,
    refine_summary_dir: str = os.path.join("Memory", "cases", "EPF", "summary_refine"),
    refine_temperature: float = 0.2,
    refine_top_p: float = 1.0,
) -> Tuple[bool, Dict[str, Any]]:
    cot_vectors, feat_vectors, entries = _load_vector_db(db_path)
    current_number: Optional[int] = None
    try:
        current_number = int(str(sample_id).split(":")[-1])
    except Exception:
        current_number = None
    if current_number is not None and entries:
        keep_idxs: List[int] = []
        for i, e in enumerate(entries):
            if not isinstance(e, dict):
                continue
            sid = e.get("id")
            try:
                n = int(str(sid).split(":")[-1])
            except Exception:
                continue
            if n <= current_number:
                keep_idxs.append(i)
        if keep_idxs:
            cot_vectors = cot_vectors[keep_idxs] if cot_vectors.size else cot_vectors
            feat_vectors = feat_vectors[keep_idxs] if feat_vectors.size else feat_vectors
            entries = [entries[i] for i in keep_idxs]
        else:
            cot_vectors = np.zeros((0, 0), dtype=np.float32)
            feat_vectors = np.zeros((0, 0), dtype=np.float32)
            entries = []
    cot_replace_th = float(cot_replace_threshold)
    feat_replace_th = float(feat_replace_threshold)
    cot_refine_th = float(cot_refine_threshold)
    feat_refine_th = float(feat_refine_threshold)

    text = _distilled_to_text(summary)
    text_hash = _stable_hash(text)
    cot_vec = _embed_text(
        text=text,
        api_key=api_key,
        base_url=embedding_base_url,
        model=embedding_model,
    )
    if cot_vec is None:
        return False, {"dedup_error": "cot_embedding_failed"}

    feat_missing = False
    try:
        if feature_vector is None:
            feat_vec = np.zeros((12,), dtype=np.float32)
            feat_missing = True
        else:
            feat_vec = np.asarray(feature_vector, dtype=np.float32).reshape(-1)
            if feat_vec.size == 0 or np.isnan(feat_vec).any():
                feat_vec = np.zeros((12,), dtype=np.float32)
                feat_missing = True
            else:
                # feat_vec = _normalize_vector(feat_vec)
                # if float(np.linalg.norm(feat_vec)) <= 0:
                #     feat_vec = np.zeros((12,), dtype=np.float32)
                #     feat_missing = True
                pass
    except Exception:
        feat_vec = np.zeros((12,), dtype=np.float32)
        feat_missing = True

    summary_file_for_entry = summary_file

    if cot_vectors.size == 0:
        cot_vectors = cot_vec.reshape(1, -1)
        feat_vectors = feat_vec.reshape(1, -1)
        entries = [{"id": sample_id, "text_hash": text_hash, "summary_file": summary_file_for_entry, "confidence": 0}]
        _save_vector_db(db_path, cot_vectors, feat_vectors, entries)
        payload = {
            "ts": datetime.now().isoformat(),
            "action": "add",
            "sample_id": sample_id,
            "db_path": db_path,
            "top1_similarity": 0.0,
            "top1_id": None,
            "top1_cot_similarity": 0.0,
            "top1_feat_similarity": 0.0,
        }
        _print_dedup_event(payload)
        _append_jsonl(log_path or "", payload)
        return False, {
            "dedup_top1_similarity": 0.0,
            "dedup_top1_cot_similarity": 0.0,
            "dedup_top1_feat_similarity": 0.0,
            "dedup_added": True,
        }

    if cot_vectors.shape[1] != cot_vec.shape[0]:
        return False, {"dedup_error": "cot_embedding_dim_mismatch"}
    if feat_vectors.ndim != 2 or feat_vectors.shape[0] != cot_vectors.shape[0]:
        feat_vectors = np.zeros((cot_vectors.shape[0], feat_vec.shape[0]), dtype=np.float32)
    if feat_vectors.shape[1] != feat_vec.shape[0]:
        feat_vectors = np.zeros((cot_vectors.shape[0], feat_vec.shape[0]), dtype=np.float32)

    cot_sims = np.dot(cot_vectors, cot_vec)
    feat_sims = np.zeros((cot_vectors.shape[0],), dtype=np.float32)

    if feat_vectors.size > 0 and feat_vectors.shape[0] > 0 and not feat_missing:
        if StandardScaler is not None and cosine_similarity is not None:
            try:
                scaler = StandardScaler()
                scaler.fit(feat_vectors)
                hist_scaled = scaler.transform(feat_vectors)
                curr_scaled = scaler.transform(feat_vec.reshape(1, -1))
                sims = cosine_similarity(hist_scaled, curr_scaled).flatten()
                feat_sims = np.nan_to_num(sims, nan=0.0)
            except Exception:
                pass
        else:
             # Fallback
             pass
    combined = np.minimum(cot_sims, feat_sims)
    top_idx = int(np.argmax(combined)) if combined.size else -1
    top_sim = float(combined[top_idx]) if top_idx >= 0 else 0.0
    top_id = entries[top_idx].get("id") if (0 <= top_idx < len(entries)) else None
    top_cot = float(cot_sims[top_idx]) if top_idx >= 0 else 0.0
    top_feat = float(feat_sims[top_idx]) if top_idx >= 0 else 0.0

    summary_update: Dict[str, Any] = {
        "dedup_top1_similarity": top_sim,
        "dedup_top1_id": top_id,
        "dedup_top1_cot_similarity": top_cot,
        "dedup_top1_feat_similarity": top_feat,
        "dedup_feature_missing": bool(feat_missing),
    }
    summary_file_for_entry = summary_file

    replace_idxs = np.where((cot_sims >= cot_replace_th) & (feat_sims >= feat_replace_th))[0].astype(int).tolist()
    if replace_idxs:
        removed_ids: List[Any] = []
        for i in replace_idxs:
            if 0 <= i < len(entries):
                removed_ids.append(entries[i].get("id"))
        keep_mask = np.ones(cot_vectors.shape[0], dtype=bool)
        keep_mask[replace_idxs] = False
        cot_vectors = cot_vectors[keep_mask]
        feat_vectors = feat_vectors[keep_mask]
        entries = [e for j, e in enumerate(entries) if keep_mask[j]]

        entries = [e for e in entries if e.get("id") != sample_id]
        if entries and cot_vectors.shape[0] != len(entries):
            n = min(cot_vectors.shape[0], len(entries))
            cot_vectors = cot_vectors[:n]
            feat_vectors = feat_vectors[:n]
            entries = entries[:n]

        cot_vectors = np.vstack([cot_vectors, cot_vec.reshape(1, -1)]) if cot_vectors.size else cot_vec.reshape(1, -1)
        feat_vectors = np.vstack([feat_vectors, feat_vec.reshape(1, -1)]) if feat_vectors.size else feat_vec.reshape(1, -1)
        entries = entries + [{"id": sample_id, "text_hash": text_hash, "summary_file": summary_file_for_entry, "confidence": 0}]
        _save_vector_db(db_path, cot_vectors, feat_vectors, entries)

        payload = {
            "ts": datetime.now().isoformat(),
            "action": "replace",
            "sample_id": sample_id,
            "db_path": db_path,
            "top1_similarity": top_sim,
            "top1_id": top_id,
            "top1_cot_similarity": top_cot,
            "top1_feat_similarity": top_feat,
            "removed_count": len(replace_idxs),
            "removed_ids": removed_ids,
        }
        _print_dedup_event(payload)
        _append_jsonl(log_path or "", payload)
        summary_update.update({"dedup_removed_count": len(replace_idxs), "dedup_removed_ids": removed_ids, "dedup_replaced": True})
        summary.update(summary_update)
        return False, summary_update

    refine_idxs = np.where((cot_sims >= cot_refine_th) & (feat_sims >= feat_refine_th))[0].astype(int).tolist()
    if refine_idxs:
        refine_idxs = sorted(refine_idxs, key=lambda i: float(combined[i]), reverse=True)
        refined_from: List[Dict[str, Any]] = []
        pieces: List[str] = []
        similar_text_count = 0
        current_dp = _extract_case_level_data_pattern(summary, current_number)
        if current_dp:
            pieces.append("### current_sample_data_pattern\n" + current_dp)
        pieces.append("### current_sample_distilled\n" + (text or ""))
        for i in refine_idxs:
            if not (0 <= i < len(entries)):
                continue
            entry = entries[i]
            sid = entry.get("id")
            spath = entry.get("summary_file")
            ssim_cot = float(cot_sims[i])
            ssim_feat = float(feat_sims[i])
            refined_from.append({"id": sid, "cot_similarity": ssim_cot, "feat_similarity": ssim_feat, "summary_file": spath})
            other = _load_summary_file(spath) if isinstance(spath, str) else None
            other_text = _distilled_to_text(other) if isinstance(other, dict) else ""
            if other_text:
                pieces.append(f"### similar_sample id={sid} cot_similarity={ssim_cot} feat_similarity={ssim_feat}\n{other_text}")
                similar_text_count += 1

        refined = None
        if similar_text_count > 0:
            combined = "\n\n".join([p for p in pieces if p and p.strip()]).strip()
            refined = _refine_distilled_with_deepseek(
                api_key=api_key,
                combined_distilled_text=combined,
                temperature=refine_temperature,
                top_p=refine_top_p,
            )
        if refined:
            summary["refined_distilled"] = refined
            summary["refined_from_similar"] = refined_from
            summary["refined_similarity_thresholds"] = {
                "cot": cot_refine_th,
                "feat": feat_refine_th,
            }
            summary["refined_timestamp"] = datetime.now().isoformat()
            refined_path = _default_refined_summary_path(
                refine_dir=refine_summary_dir,
                data_name=os.path.basename(os.path.dirname(db_path)),
                sample_id=sample_id,
                original_summary_file=summary_file,
            )
            if _save_refined_summary(refined_path, summary):
                summary["refined_summary_file"] = refined_path
                summary_file_for_entry = refined_path
            text = _distilled_to_text(summary)
            text_hash = _stable_hash(text)
            cot_vec2 = _embed_text(
                text=text,
                api_key=api_key,
                base_url=embedding_base_url,
                model=embedding_model,
            )
            if cot_vec2 is not None and cot_vec2.shape == cot_vec.shape:
                cot_vec = cot_vec2
            summary_update.update({"dedup_refined": True, "dedup_refined_from_count": len(refined_from)})
        else:
            summary_update.update(
                {
                    "dedup_refined": False,
                    "dedup_refined_error": "no_similar_summary_files" if similar_text_count == 0 else "refine_failed",
                }
            )

    entries = [e for e in entries if e.get("id") != sample_id]
    if entries and cot_vectors.shape[0] != len(entries):
        n = min(cot_vectors.shape[0], len(entries))
        cot_vectors = cot_vectors[:n]
        feat_vectors = feat_vectors[:n]
        entries = entries[:n]

    cot_vectors = np.vstack([cot_vectors, cot_vec.reshape(1, -1)]) if cot_vectors.size else cot_vec.reshape(1, -1)
    feat_vectors = np.vstack([feat_vectors, feat_vec.reshape(1, -1)]) if feat_vectors.size else feat_vec.reshape(1, -1)
    entries = entries + [{"id": sample_id, "text_hash": text_hash, "summary_file": summary_file_for_entry, "confidence": 0}]
    _save_vector_db(db_path, cot_vectors, feat_vectors, entries)

    action = "refine_add" if refine_idxs and summary_update.get("dedup_refined") else "add"
    payload = {
        "ts": datetime.now().isoformat(),
        "action": action,
        "sample_id": sample_id,
        "db_path": db_path,
        "top1_similarity": top_sim,
        "top1_id": top_id,
        "top1_cot_similarity": top_cot,
        "top1_feat_similarity": top_feat,
    }
    if action == "refine_add":
        payload.update(
            {
                "refined_from": summary.get("refined_from_similar"),
                "refined_count": len(summary.get("refined_distilled") or []),
            }
        )
    _print_dedup_event(payload)
    _append_jsonl(log_path or "", payload)
    summary_update.update({"dedup_added": True})
    summary.update(summary_update)
    return False, summary_update

def _build_case_query(data_name: str, sample: Dict[str, Any], dataset_columns: List[str]) -> str:
    target = sample.get("file_attr")
    look_back = sample.get("file_look_back")
    pred_window = sample.get("file_pred_window")
    index = sample.get("file_index")
    covariates = [c for c in ["Grid load forecast", "Wind power forecast"] if c in set(dataset_columns)]
    covariates_str = ", ".join(covariates) if covariates else "N/A"
    return (
        f"Dataset: {data_name}\n"
        f"Target: {target}\n"
        f"Look-back: {look_back}\n"
        f"Prediction window: {pred_window}\n"
        f"Sample index: {index}\n"
        f"Covariates: {covariates_str}"
    )

def distill_good_bad_cases_with_deepseek(
    data_name: str,
    api_key: str,
    quantile: float = 0.4,
    attr: str = None,
    look_back: int = None,
    pred_window: int = None,
    number: int = None,
    dataset_dir: str = "datasets",
    train_ratio: float = 0.8,
    slide_window: int = 24,
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_cases_per_group: Optional[int] = None,
    prompt_output_dir: str = os.path.join("Memory", "cases", "origin", "EPF", "prompt_cat"),
) -> Dict[str, Any]:
    data = _load_dataset(data_name=data_name, dataset_dir=dataset_dir, train_ratio=train_ratio)
    dataset_columns = [c for c in data.columns]

    split_result = split_good_bad_cases_by_mse(
        data_name=data_name,
        quantile=quantile,
        attr=attr,
        look_back=look_back,
        pred_window=pred_window,
        number=number,
        dataset_dir=dataset_dir,
        train_ratio=train_ratio,
        slide_window=slide_window,
    )

    good_cases: List[Dict[str, Any]] = split_result.get("good_cases", [])
    bad_cases: List[Dict[str, Any]] = split_result.get("bad_cases", [])

    if number is not None:
        good_cases = [c for c in good_cases if c.get("file_index") == number]
        bad_cases = [c for c in bad_cases if c.get("file_index") == number]

    if max_cases_per_group is not None:
        good_cases = good_cases[: max(0, int(max_cases_per_group))]
        bad_cases = bad_cases[: max(0, int(max_cases_per_group))]

    model = deepseek_api_output(api_key=api_key, temperature=temperature, top_p=top_p)

    def distill_one(sample: Dict[str, Any], case_type: str) -> Dict[str, Any]:
        query = _build_case_query(data_name=data_name, sample=sample, dataset_columns=dataset_columns)
        step_sequence = (sample.get("reasoning") or "").strip()
        answer_text = sample.get("answer") or ""
        forecast_lines, _ = _extract_forecast_lines(answer_text)
        if forecast_lines:
            prediction = "\n".join(forecast_lines)
        else:
            prediction = answer_text.strip()

        file_attr = sample.get("file_attr")
        file_look_back = sample.get("file_look_back")
        file_pred_window = sample.get("file_pred_window")
        file_index = sample.get("file_index")

        cov_text = _get_covariate_window_text(
            data=data,
            covariates=["Grid load forecast", "Wind power forecast"],
            look_back=int(file_look_back),
            pred_window=int(file_pred_window),
            number=int(file_index),
            slide_window=slide_window,
        )
        if cov_text:
            query = f"{query}\n\n{cov_text}"

        actual_lines = _get_ground_truth_lines(
            data=data,
            attr=file_attr,
            look_back=int(file_look_back),
            pred_window=int(file_pred_window),
            number=int(file_index),
            slide_window=slide_window,
        )
        actual_outcome = "\n".join(actual_lines) if actual_lines else "N/A"

        prompt: str
        if case_type == "good":
            prompt = time_series_memory_extraction_prompt.format(
                query=query[:12000],
                step_sequence=step_sequence[:12000],
                prediction=prediction[:12000],
                outcome="successful",
                actual_outcome=actual_outcome[:12000],
            )
        else:
            prompt = time_series_failure_extraction_prompt.format(
                query=query[:12000],
                step_sequence=step_sequence[:12000],
                prediction=prediction[:12000],
                actual_outcome=actual_outcome[:12000],
            )

        try:
            os.makedirs(prompt_output_dir, exist_ok=True)
            prompt_file = os.path.join(
                prompt_output_dir,
                f"prompt_{case_type}_{data_name}_{file_attr}_{file_look_back}_{file_pred_window}_{file_index}.txt",
            )
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt)
        except Exception:
            pass

        _, output_text = model(prompt)
        cleaned = _clean_json_text(output_text)
        try:
            parsed = json.loads(cleaned)
            data_pattern, distilled = _lift_data_pattern_from_distilled(parsed)
            if data_pattern is not None:
                print(data_pattern)
                _save_data_pattern(data_name, file_attr, file_look_back, file_pred_window, file_index, data_pattern, case_type)
            return {
                "case_type": case_type,
                "file_attr": file_attr,
                "file_look_back": file_look_back,
                "file_pred_window": file_pred_window,
                "file_index": file_index,
                "mse": sample.get("mse"),
                "mae": sample.get("mae"),
                "rmse": sample.get("rmse"),
                "distilled": distilled,
            }
        except Exception:
            return {
                "case_type": case_type,
                "file_attr": file_attr,
                "file_look_back": file_look_back,
                "file_pred_window": file_pred_window,
                "file_index": file_index,
                "mse": sample.get("mse"),
                "mae": sample.get("mae"),
                "rmse": sample.get("rmse"),
                "error": "Failed to parse JSON",
                "raw_response": output_text,
            }

    good_distilled = [distill_one(s, "good") for s in good_cases]
    bad_distilled = [distill_one(s, "bad") for s in bad_cases]

    return {
        "data_name": data_name,
        "quantile": split_result.get("quantile"),
        "threshold": split_result.get("threshold"),
        "evaluated_count": split_result.get("evaluated_count", 0),
        "good_count": split_result.get("good_count", 0),
        "bad_count": split_result.get("bad_count", 0),
        "skipped_count": split_result.get("skipped_count", 0),
        "analysis_timestamp": datetime.now().isoformat(),
        "good_cases_distilled": good_distilled,
        "bad_cases_distilled": bad_distilled,
        "skipped_cases": split_result.get("skipped_cases", []),
    }

def split_good_bad_cases_by_mse(
    data_name: str,
    quantile: float = 0.4,
    attr: str = None,
    look_back: int = None,
    pred_window: int = None,
    number: int = None,
    dataset_dir: str = "datasets",
    train_ratio: float = 0.8,
    slide_window: int = 24,
) -> Dict[str, Any]:
    data = _load_dataset(data_name=data_name, dataset_dir=dataset_dir, train_ratio=train_ratio)
    samples = load_memory_samples(data_name=data_name, attr=attr, look_back=look_back, pred_window=pred_window, number=None)
    if number is not None:
        try:
            upper = int(number)
            samples = [s for s in samples if isinstance(s, dict) and isinstance(s.get("file_index"), int) and int(s.get("file_index")) <= upper]
        except Exception:
            pass

    evaluated = []
    skipped = []

    for s in samples:
        file_attr = s.get("file_attr")
        file_look_back = s.get("file_look_back")
        file_pred_window = s.get("file_pred_window")
        file_index = s.get("file_index")
        answer_text = s.get("answer", "")

        true_values = _get_ground_truth(
            data=data,
            attr=file_attr,
            look_back=file_look_back,
            pred_window=file_pred_window,
            number=file_index,
            slide_window=slide_window,
        )
        if not true_values:
            skipped.append({**s, "skip_reason": "missing_ground_truth"})
            continue

        predicted_values = []
        parsed_prediction = s.get("parsed_prediction")
        if isinstance(parsed_prediction, list) and parsed_prediction:
            try:
                vals = [float(v) for v in parsed_prediction]
                if vals and not any(np.isnan(v) for v in vals):
                    predicted_values = vals
            except Exception:
                predicted_values = []

        if not predicted_values:
            predicted_values = get_result(answer_text)
        if not predicted_values:
            skipped.append({**s, "skip_reason": "missing_prediction"})
            continue

        mse, mae, rmse = evaluate_prediction_quality(predicted_values, true_values)
        if mse is None:
            skipped.append({**s, "skip_reason": "invalid_metrics"})
            continue

        evaluated.append(
            {
                **s,
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "pred_length": int(min(len(predicted_values), len(true_values))),
                "true_length": int(len(true_values)),
            }
        )

    mses = [e["mse"] for e in evaluated]
    if not mses:
        return {
            "data_name": data_name,
            "quantile": quantile,
            "threshold": None,
            "evaluated_cases": [],
            "good_cases": [],
            "bad_cases": [],
            "skipped_cases": skipped,
        }

    threshold = float(np.quantile(np.asarray(mses, dtype=float), quantile))
    good_cases = [e for e in evaluated if e["mse"] <= threshold]
    bad_cases = [e for e in evaluated if e["mse"] > threshold]

    return {
        "data_name": data_name,
        "quantile": quantile,
        "threshold": threshold,
        "evaluated_count": len(evaluated),
        "good_count": len(good_cases),
        "bad_count": len(bad_cases),
        "good_cases": good_cases,
        "bad_cases": bad_cases,
        "skipped_count": len(skipped),
        "skipped_cases": skipped,
    }

def load_memory_samples(data_name: str, attr: str = None, look_back: int = None, pred_window: int = None, number: int = None) -> List[Dict[str, Any]]:
    """
    从Memory目录中加载样本数据
    
    Args:
        data_name: 数据集名称，如'EPF-NP'
        attr: 属性名称，如'OT'，可选
        look_back: 回看窗口大小，可选
        pred_window: 预测窗口大小，可选
    
    Returns:
        List[Dict]: 包含样本数据的列表，每个字典包含index, reasoning, answer等字段
    """
    memory_dir = f'Memory/cases/origin/EPF/{data_name}'
    if not os.path.exists(memory_dir):
        print(f"Warning: Memory directory {memory_dir} not found!")
        return []
    
    samples = []
    result_files = []
    
    # 获取所有相关的结果文件
    for file in os.listdir(memory_dir):
        if file.endswith('.json'):
            if attr and look_back and pred_window:
                base_prefix = f"result_{attr}_{data_name}_{look_back}_{pred_window}_"
                if number is not None:
                    if file == f"{base_prefix}{number}.json":
                        result_files.append(file)
                else:
                    if file.startswith(base_prefix):
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
                    sample_index = sample.get('index', 0)
                    if isinstance(sample_index, str) and sample_index.strip().lower() == "average":
                        continue
                    sample_info = {
                        'file_name': file,
                        'file_attr': file_attr,
                        'file_look_back': file_look_back,
                        'file_pred_window': file_pred_window,
                        'file_index': file_index,
                        'sample_index': sample_index,
                        'reasoning': sample.get('reasoning', ''),
                        'answer': sample.get('answer', ''),
                        'parsed_prediction': sample.get('parsed_prediction', None),
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
    if not summary:
        print("Invalid or empty summary provided.")
        return

    if summary.get("error"):
        print("Invalid or empty summary provided.")
        print(f"An error occurred during analysis: {summary['error']}")
        if summary.get("raw_response"):
            print(f"Raw response: {summary['raw_response']}")
        return

    if "overall_summary" in summary:
        print("\n" + "=" * 80)
        print("REASONING TRAJECTORY SUMMARY (Generated by DeepSeek)")
        print("=" * 80)

        print(f"\nTotal Samples Analyzed: {summary.get('total_samples_analyzed', 'N/A')}")
        print(f"Analysis Timestamp: {summary.get('analysis_timestamp', 'N/A')}")

        print("\n📜 OVERALL SUMMARY:")
        print(f"  {summary.get('overall_summary', 'Not available.')}")

        print("\n📊 METHOD DISTRIBUTION:")
        method_dist = summary.get("method_distribution", {})
        if method_dist:
            for method, percentage in method_dist.items():
                print(f"  - {method}: {percentage}")
        else:
            print("  Not available.")

        print("\n📈 COMMON DATA PATTERNS:")
        patterns = summary.get("common_data_patterns", [])
        if patterns:
            for pattern in patterns:
                print(f"  - {pattern}")
        else:
            print("  Not available.")

        print("\n💡 KEY INSIGHTS:")
        insights = summary.get("key_insights", [])
        if insights:
            for insight in insights:
                print(f"  - {insight}")
        else:
            print("  Not available.")

        print("\n🧠 COMPLEXITY ANALYSIS:")
        print(f"  {summary.get('complexity_analysis', 'Not available.')}")

        print("\n" + "=" * 80)
        return

    if "good_cases_distilled" in summary or "bad_cases_distilled" in summary:
        def _case_id(item: Dict[str, Any]) -> str:
            return (
                f"{item.get('file_attr')}_lb{item.get('file_look_back')}"
                f"_pw{item.get('file_pred_window')}_idx{item.get('file_index')}"
            )

        def _extract_first_dict(x: Any) -> Optional[Dict[str, Any]]:
            if isinstance(x, list) and x and isinstance(x[0], dict):
                return x[0]
            if isinstance(x, dict):
                return x
            return None

        def _print_group(title: str, items: List[Dict[str, Any]], kind: str, max_show: int = 5):
            print(f"\n{title}: {len(items)}")
            shown = 0
            for item in items:
                if shown >= max_show:
                    break
                shown += 1
                prefix = f"  - {_case_id(item)}"
                if item.get("error"):
                    print(f"{prefix} | error={item.get('error')}")
                    continue
                distilled_first = _extract_first_dict(item.get("distilled"))
                if not distilled_first:
                    print(f"{prefix} | distilled=N/A")
                    continue
                if kind == "good":
                    rule = distilled_first.get("success_insight") or distilled_first.get("experience") or "N/A"
                    conf = distilled_first.get("confidence", "N/A")
                else:
                    rule = distilled_first.get("preventative_rule") or distilled_first.get("failure_analysis") or "N/A"
                    conf = distilled_first.get("remedy_confidence", "N/A")
                tags = distilled_first.get("tags", [])
                tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
                print(f"{prefix} | conf={conf} | tags={tags_str}")
                print(f"    {rule}")

        print("\n" + "=" * 80)
        print("GOOD/BAD CASE DISTILLATION SUMMARY (Generated by DeepSeek)")
        print("=" * 80)

        print(f"\nDataset: {summary.get('data_name', 'N/A')}")
        print(f"Analysis Timestamp: {summary.get('analysis_timestamp', 'N/A')}")
        print(f"Quantile: {summary.get('quantile', 'N/A')}")
        print(f"Threshold (MSE): {summary.get('threshold', 'N/A')}")
        print(
            f"Evaluated: {summary.get('evaluated_count', 'N/A')}, "
            f"Good: {summary.get('good_count', 'N/A')}, "
            f"Bad: {summary.get('bad_count', 'N/A')}, "
            f"Skipped: {summary.get('skipped_count', 'N/A')}"
        )

        _print_group("Good Cases Distilled", summary.get("good_cases_distilled", []) or [], "good")
        _print_group("Bad Cases Distilled", summary.get("bad_cases_distilled", []) or [], "bad")

        print("\n" + "=" * 80)
        return

    print("Invalid or empty summary provided.")


def analyze_memory_reasoning(data_name: str, api_key: str, attr: str = None, look_back: int = None,  
                             pred_window: int = None, output_file: str = None, output_dir: str = None,  
                             print_report: bool = True, temperature: float = 0.2, top_p: float = 1.0,
                             overwrite: bool = False, number: int = None, quantile: float = 0.4,
                             distill_by_case: bool = True,
                             max_cases_per_group: Optional[int] = None,
                             dataset_dir: str = "datasets",
                             train_ratio: float = 0.8,
                             slide_window: int = 24,
                             dedup_enable: bool = True,
                             dedup_similarity_threshold: float = 0.9,
                             vector_db_dir: str = os.path.join("Memory", "cases", "EPF", "vector_db"),
                             dedup_log_file: Optional[str] = None,
                             debug_trace: bool = False,
                             infer_mode: bool = False,
                             update_confidence:int = 1,
                             infer_similar_ids: Optional[List[str]] = None,
                             embedding_base_url: Optional[str] = None,
                             embedding_model: str = "text-embedding-3-small") -> Dict[str, Any]:
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
    infer_ids = infer_similar_ids if isinstance(infer_similar_ids, list) else []
    if embedding_base_url is None:
        embedding_base_url = os.getenv("OPENAI_BASE_URL", "https://api2.aigcbest.top/v1")
    if debug_trace:
        print("=" * 80)
        print("[TRACE] analyze_memory_reasoning start")
        print(f"[TRACE] data_name={data_name} attr={attr} look_back={look_back} pred_window={pred_window} number={number}")
        print(f"[TRACE] output_file={output_file} output_dir={output_dir} overwrite={overwrite}")
        print(f"[TRACE] distill_by_case={distill_by_case} quantile={quantile} max_cases_per_group={max_cases_per_group}")
        print(f"[TRACE] dataset_dir={dataset_dir} train_ratio={train_ratio} slide_window={slide_window}")
        print(f"[TRACE] dedup_enable={dedup_enable} dedup_similarity_threshold={dedup_similarity_threshold}")
        print(f"[TRACE] vector_db_dir={vector_db_dir} dedup_log_file={dedup_log_file}")
        print(f"[TRACE] infer_mode={infer_mode} infer_similar_ids_count={len(infer_ids)}")
        print(f"[TRACE] embedding_base_url={embedding_base_url} embedding_model={embedding_model}")
        print(f"[TRACE] OpenAI_available={OpenAI is not None} HashingVectorizer_available={HashingVectorizer is not None}")
        print("=" * 80)

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
    if debug_trace:
        print(f"[TRACE] resolved_output_file={output_file}")

    # ==== [2] 如果文件存在且未允许覆盖，直接跳过推理 ====
    if output_file and os.path.exists(output_file) and not overwrite:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_result = json.load(f)

            # 判断是否完整（例如 len(existing_result) == 1 代表任务已完成）
            if isinstance(existing_result, dict) and (
                "overall_summary" in existing_result
                or "good_cases_distilled" in existing_result
                or "bad_cases_distilled" in existing_result
            ):
                print(f"Existing result found at {output_file}. Task already done, skipping API call.")
                if (not infer_mode) and dedup_enable and ("good_cases_distilled" in existing_result or "bad_cases_distilled" in existing_result):
                    db_path = os.path.join(vector_db_dir, data_name, f"db_{attr}_{look_back}_{pred_window}.npz")
                    log_path = dedup_log_file or os.path.join(vector_db_dir, data_name, f"dedup_{attr}_{look_back}_{pred_window}.jsonl")
                    sample_id = f"{data_name}:{attr}:{look_back}:{pred_window}:{number}"
                    db_exists = os.path.exists(db_path) and os.path.exists(os.path.splitext(db_path)[0] + ".json")
                    _, _, entries = _load_vector_db(db_path)
                    has_sample = any(isinstance(e, dict) and e.get("id") == sample_id for e in (entries or []))
                    if debug_trace:
                        print(f"[TRACE] existing_summary_dedup db_path={db_path}")
                        print(f"[TRACE] existing_summary_dedup log_path={log_path}")
                        print(f"[TRACE] existing_summary_dedup sample_id={sample_id}")
                        print(f"[TRACE] existing_summary_dedup db_exists={db_exists} entry_count={len(entries or [])} has_sample={has_sample}")
                    if (not db_exists) or (not has_sample):
                        feature_vector = _extract_feature_vector(
                            data_name=data_name,
                            attr=str(attr),
                            look_back=int(look_back),
                            number=int(number),
                            dataset_dir=dataset_dir,
                            train_ratio=float(train_ratio),
                            slide_window=int(slide_window),
                        )
                        _, dedup_info = _dedup_check_and_update_db(
                            summary=existing_result,
                            api_key=api_key,
                            db_path=db_path,
                            sample_id=sample_id,
                            cot_replace_threshold=dedup_similarity_threshold,
                            feat_replace_threshold=dedup_similarity_threshold,
                            cot_refine_threshold=0.8,
                            feat_refine_threshold=0.9,
                            embedding_base_url=embedding_base_url,
                            embedding_model=embedding_model,
                            feature_vector=feature_vector,
                            log_path=log_path,
                            summary_file=output_file,
                            refine_temperature=temperature,
                            refine_top_p=top_p,
                        )
                        if debug_trace:
                            txt = _distilled_to_text(existing_result)
                            print(f"[TRACE] existing_summary_dedup distilled_text_len={len(txt or '')}")
                            print(f"[TRACE] existing_summary_dedup dedup_info={dedup_info}")
                        if isinstance(dedup_info, dict) and dedup_info:
                            existing_result.update(dedup_info)
                            try:
                                with open(output_file, "w", encoding="utf-8") as f:
                                    json.dump(existing_result, f, indent=2, ensure_ascii=False)
                            except Exception:
                                pass
                if infer_mode and infer_ids and int(update_confidence) == 1:
                    if _is_good_case_summary(existing_result, number):
                        meta_path = os.path.join(vector_db_dir, data_name, f"db_{attr}_{look_back}_{pred_window}.json")
                        _increase_entry_confidence(meta_path, infer_ids, 0.01)
                    else:
                        print(f"[CONF] skip not_good_case_summary number={number} infer_ids={len(infer_ids)}")
                if print_report:
                    if "overall_summary" in existing_result:
                        print_deepseek_summary(existing_result)
                return existing_result
            else:
                print(f"Existing file found but incomplete, continuing to summarize reasoning...")
        except Exception as e:
            print(f"Error reading existing result file: {e}. Will regenerate summary.")

    if infer_mode:
        if output_file and os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_result = json.load(f)
                if isinstance(existing_result, dict) and int(update_confidence) == 1 and infer_ids and _is_good_case_summary(existing_result, number):
                    meta_path = os.path.join(vector_db_dir, data_name, f"db_{attr}_{look_back}_{pred_window}.json")
                    _increase_entry_confidence(meta_path, infer_ids, 0.01)
                return existing_result if isinstance(existing_result, dict) else {}
            except Exception:
                pass
        return {}

    print(f"Loading samples from Memory for {data_name}...")

    samples = load_memory_samples(data_name, attr, look_back, pred_window, number)
    if not samples:
        print("No samples found!")
        return {}
    if debug_trace:
        uniq_files = sorted({s.get("file_name") for s in samples if isinstance(s, dict) and s.get("file_name")})
        uniq_idxs = sorted({s.get("file_index") for s in samples if isinstance(s, dict) and s.get("file_index") is not None})
        print(f"[TRACE] loaded_samples count={len(samples)} unique_files={len(uniq_files)} unique_file_index={len(uniq_idxs)}")
        if uniq_idxs:
            print(f"[TRACE] file_index_minmax=({min(uniq_idxs)},{max(uniq_idxs)})")
        first = samples[0] if samples else None
        if isinstance(first, dict):
            print(f"[TRACE] first_sample_keys={sorted(list(first.keys()))}")
            print(f"[TRACE] first_sample file_name={first.get('file_name')} file_index={first.get('file_index')} sample_index={first.get('sample_index')}")

    # ==== [3] 若未完成任务，则调用 DeepSeek API ====
    if distill_by_case:
        summary = distill_good_bad_cases_with_deepseek(
            data_name=data_name,
            api_key=api_key,
            quantile=quantile,
            attr=attr,
            look_back=look_back,
            pred_window=pred_window,
            number=number,
            dataset_dir=dataset_dir,
            train_ratio=train_ratio,
            slide_window=slide_window,
            temperature=temperature,
            top_p=top_p,
            max_cases_per_group=max_cases_per_group,
        )
    else:
        summary = summarize_reasoning_with_deepseek(samples, api_key, temperature, top_p)
    if debug_trace:
        if isinstance(summary, dict):
            keys = sorted(list(summary.keys()))
            print(f"[TRACE] summary_keys={keys}")
            if "evaluated_count" in summary or "good_count" in summary or "bad_count" in summary or "skipped_count" in summary:
                print(
                    f"[TRACE] evaluated_count={summary.get('evaluated_count')} good_count={summary.get('good_count')} "
                    f"bad_count={summary.get('bad_count')} skipped_count={summary.get('skipped_count')}"
                )
            skipped = summary.get("skipped_cases") if isinstance(summary, dict) else None
            if isinstance(skipped, list) and skipped:
                reasons = [s.get("skip_reason") for s in skipped if isinstance(s, dict)]
                reason_counts = Counter([r for r in reasons if r])
                print(f"[TRACE] skipped_reason_counts={dict(reason_counts)}")
                for item in skipped[:5]:
                    if not isinstance(item, dict):
                        continue
                    print(
                        f"[TRACE] skipped_case file={item.get('file_name')} file_index={item.get('file_index')} "
                        f"sample_index={item.get('sample_index')} reason={item.get('skip_reason')}"
                    )
            txt = _distilled_to_text(summary) if isinstance(summary, dict) else ""
            print(f"[TRACE] distilled_text_len={len(txt or '')}")
    if isinstance(summary, dict):
        distilled_text = _distilled_to_text(summary)
        if not (distilled_text and distilled_text.strip()):
            chunks: List[str] = []
            for s in (samples or [])[:6]:
                if not isinstance(s, dict):
                    continue
                reasoning = (s.get("reasoning") or "").strip()
                answer = (s.get("answer") or "").strip()
                parsed_pred = s.get("parsed_prediction")
                if isinstance(parsed_pred, list) and parsed_pred:
                    parsed_str = json.dumps(parsed_pred[: min(10, len(parsed_pred))], ensure_ascii=False)
                else:
                    parsed_str = ""
                piece = "\n".join([p for p in [reasoning, answer, parsed_str] if p]).strip()
                if piece:
                    chunks.append(piece[:4000])
            fallback_text = "\n\n---\n\n".join(chunks).strip()
            if fallback_text:
                summary["refined_distilled"] = [
                    {
                        "experience": "fallback_from_raw_reasoning_and_answer",
                        "success_insight": fallback_text,
                        "meta_logic": {
                            "source": "fallback",
                            "has_ground_truth": False,
                            "file_attr": attr,
                            "look_back": look_back,
                            "pred_window": pred_window,
                            "number": number,
                        },
                    }
                ]
                try:
                    prompt_output_dir = os.path.join("Memory", "cases", "origin", "EPF", "prompt_cat")
                    os.makedirs(prompt_output_dir, exist_ok=True)
                    prompt_file = os.path.join(
                        prompt_output_dir,
                        f"prompt_fallback_{data_name}_{attr}_{look_back}_{pred_window}_{number}.txt",
                    )
                    with open(prompt_file, "w", encoding="utf-8") as f:
                        f.write(fallback_text)
                    if debug_trace:
                        print(f"[TRACE] wrote_fallback_prompt={prompt_file}")
                except Exception:
                    pass

    # ==== [4] 保存结果 ====
    if infer_mode and infer_ids and int(update_confidence) == 1:
        if _is_good_case_summary(summary, number):
            meta_path = os.path.join(vector_db_dir, data_name, f"db_{attr}_{look_back}_{pred_window}.json")
            _increase_entry_confidence(meta_path, infer_ids, 0.01)
        else:
            print(f"[CONF] skip not_good_case_summary number={number} infer_ids={len(infer_ids)}")

    discarded_by_dedup = False
    dedup_info: Dict[str, Any] = {}
    if (not infer_mode) and dedup_enable and isinstance(summary, dict) and ("good_cases_distilled" in summary or "bad_cases_distilled" in summary):
        db_path = os.path.join(vector_db_dir, data_name, f"db_{attr}_{look_back}_{pred_window}.npz")
        log_path = dedup_log_file or os.path.join(vector_db_dir, data_name, f"dedup_{attr}_{look_back}_{pred_window}.jsonl")
        sample_id = f"{data_name}:{attr}:{look_back}:{pred_window}:{number}"
        if debug_trace:
            print(f"[TRACE] dedup db_path={db_path}")
            print(f"[TRACE] dedup log_path={log_path}")
            print(f"[TRACE] dedup sample_id={sample_id}")
        feature_vector = _extract_feature_vector(
            data_name=data_name,
            attr=str(attr),
            look_back=int(look_back),
            number=int(number),
            dataset_dir=dataset_dir,
            train_ratio=float(train_ratio),
            slide_window=int(slide_window),
        )
        discarded_by_dedup, dedup_info = _dedup_check_and_update_db(
            summary=summary,
            api_key=api_key,
            db_path=db_path,
            sample_id=sample_id,
            cot_replace_threshold=dedup_similarity_threshold,
            feat_replace_threshold=dedup_similarity_threshold,
            cot_refine_threshold=0.8,
            feat_refine_threshold=0.9,
            embedding_base_url=embedding_base_url,
            embedding_model=embedding_model,
            feature_vector=feature_vector,
            log_path=log_path,
            summary_file=output_file,
            refine_temperature=temperature,
            refine_top_p=top_p,
        )
        summary.update(dedup_info)
        if debug_trace:
            print(f"[TRACE] dedup_info={dedup_info}")
            db_exists = os.path.exists(db_path) and os.path.exists(os.path.splitext(db_path)[0] + ".json")
            _, _, entries2 = _load_vector_db(db_path)
            print(f"[TRACE] dedup_after db_exists={db_exists} entry_count={len(entries2 or [])}")

    if (not infer_mode) and output_file:
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
    if print_report and isinstance(summary, dict) and "overall_summary" in summary:
        print_deepseek_summary(summary)

    return summary



def example_usage():

    api_key = os.getenv("OPENAI_API_KEY")


    print("=== 思考轨迹分析功能示例 (使用 DeepSeek API) ===\n")
    
    # 示例1: 分析EPF-NP数据集的所有样本
    print("1. Analyzing all samples for EPF-NP...")
    analyze_memory_reasoning(
        data_name='EPF-NP',
        api_key=api_key,
        attr='OT',
        look_back=96,
        pred_window=96,
        output_dir='/data/songliv/TS/TimeReasoner/Memory/summary_outputs',
        overwrite=False,
        number=1
    )


if __name__ == "__main__":
    result = split_good_bad_cases_by_mse(data_name="EPF-NP", quantile=0.4)
    print(
        f"[{result['data_name']}] evaluated={result.get('evaluated_count', 0)}, "
        f"skipped={result.get('skipped_count', 0)}, "
        f"threshold(q={result['quantile']})={result.get('threshold')}, "
        f"good={result.get('good_count', 0)}, bad={result.get('bad_count', 0)}"
    )

