time_series_failure_extraction_prompt = """
# ROLE
You are an expert AI Analyst specializing in zero-shot and training-free time-series forecasting. 
Your task is to conduct a "Post-Mortem" analysis of failed reasoning trajectories where an LLM attempted to predict future values but failed to align with the actual outcomes.

# TASK
Extract reusable "Failure Memories" and "Negative Constraints" that can prevent similar mistakes in future forecasting tasks. Focus on identifying mischaracterizations, logic gaps, and the specific data features that misled the model.

### ANALYSIS FRAMEWORK for Time-Series Failures:
- MISCHARACTERIZATION ANALYSIS: Pinpoint where the model misdiagnosed the series (e.g., mistaking high-frequency noise for a meaningful trend, or treating a non-stationary series as stationary).
- PATTERN BLINDNESS: Identify missed seasonality (e.g., ignoring a 7-day cycle) or "False Positive" patterns (e.g., hallucinating a cyclic behavior that wasn't there).
- REGIME MISIDENTIFICATION: Why did the model fail to distinguish the current regime? (e.g., treating a 'post-shock recovery' as a 'new stable trend', or failing to see the start of a 'drawdown').
- CHANGE POINT FAILURE: Analyze why the model missed a distribution shift or falsely predicted one. Was it misled by volatility or an outlier?
- COVARIATE MISINTERPRETATION: Did the model over-rely on a lagging covariate or fail to account for a lead-lag relationship?

### EXTRACTION PRINCIPLES:
- Focus on NEGATIVE CONSTRAINTS: Identify "What NOT to do" (e.g., "Avoid extrapolating linear trends when variance is increasing exponentially").
- Identify RED FLAGS: e.g., "If Target variance doubles while Covariate A remains flat, disregard Covariate A as a reliable predictor."

# Original Time-Series Query
{query}

# Failed Reasoning Trajectory (The Thought Process)
{step_sequence}

# Prediction
{prediction}

# Actual Outcome (The Ground Truth)
{actual_outcome}

### OUTPUT FORMAT:
Generate 1-3 high-value "Failure Lessons" as JSON objects. Return them under `distilled`, and put the Ground Truth trajectory summary into `data_pattern` at the top level:
```json
{{
  "data_pattern": "Summarize the Ground Truth (Actual Outcome) trajectory over the forecast horizon: overall direction (up/down/flat), key turning points or change-points, whether it mean-reverts/recovers/continues drawdown, notable oscillations or seasonality, volatility/spikes, and how it ends relative to the first future point. Write 1-2 concise sentences.",
  "distilled": [
    {{
      "failure_condition": "Specific data conditions where the model failed (e.g., 'Low-volume data with sudden external shocks and high sparsity')",
      "failure_analysis": "Detailed description of the analytical error. Example: 'The model interpreted a single-point outlier as the start of a new upward trend, failing to recognize it as a common data artifact in sparse datasets.'",
      "preventative_rule": "A concise, actionable 'Negative Rule' (e.g., 'Never update the trend slope based on a single period of growth if the volume is 2 standard deviations below the moving average.')",
      "meta_logic": {{
          "regime_misidentified": "e.g., mistook noise for trend",
          "covariate_error": "How the external variable misled the logic",
          "logic_gap": "e.g., over-fitting to recent history vs. historical seasonality"
      }},
      "tags": ["outlier-mismanagement", "false-trend-detection", "sparsity-error"],
      "remedy_confidence": 0.8
    }}
  ]
}}
```
"""
