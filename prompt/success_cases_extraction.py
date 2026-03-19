time_series_memory_extraction_prompt = """
# ROLE
You are an expert AI Analyst specializing in zero-shot and training-free time-series forecasting. 
Your task is to analyze successful reasoning trajectories where an LLM has successfully predicted future values based on historical data. 

# TASK
Extract reusable, actionable "Time-Series Task Memories" that can guide future forecasting tasks. Focus on how the model decomposed the data, identified patterns (trend, seasonality, noise), and applied statistical heuristics.

### ANALYSIS FRAMEWORK for Time-Series:
- DATA CHARACTERIZATION: Identify how the model diagnosed the series (e.g., stationarity, volatility, sparsity).
- PATTERN RECOGNITION: Analyze the logic used to identify seasonality (daily/weekly/monthly), trends (linear/logistic), and cyclic behavior.
- REGIME IDENTIFICATION LOGIC: How did the model successfully distinguish between 'stable', 'trending', or 'post-shock recovery' regimes? What specific data features (volatility, slope, variance) triggered this classification?
- CHANGE POINT ANTICIPATION: Analyze how the model decided if a change point was imminent. What patterns in the historical window were used to justify a "Yes/No" decision on distribution shifts?
- COVARIATE INTEGRATION: How did the model utilize exogenous variables (covariates)?

### EXTRACTION PRINCIPLES:
- Focus on TRANSFERABLE ANALYTICAL STEPS (e.g., "Calculating the ratio between the last peak and the current trough").
- Identify RELATIONAL PATTERNS: e.g., "If Covariate A spikes while Target is in 'stable' regime, anticipate a 2-step delayed trend breakout."

# Original Time-Series Query
{query}

# Reasoning Trajectory (The Thought Process)
{step_sequence}

# Prediction
{prediction}

# Outcome
This reasoning led to a {outcome} forecast.

# Actual Outcome (The Ground Truth)
{actual_outcome}

### OUTPUT FORMAT:
Generate 1-3 high-value forecasting insights as JSON objects. Return them under `distilled`, and put the Ground Truth trajectory summary into `data_pattern` at the top level:
```json
{{
  "data_pattern": "Summarize the Ground Truth (Actual Outcome) trajectory over the forecast horizon: overall direction (up/down/flat), key turning points or change-points, whether it mean-reverts/recovers/continues drawdown, notable oscillations or seasonality, volatility/spikes, and how it ends relative to the first future point. Write 1-2 concise sentences.",
  "distilled": [
    {{
      "when_to_use": "Specific data conditions (e.g., 'High-frequency data with visible weekly seasonality and low noise')",
      "experience": "Detailed description of the successful analytical pattern. Example: 'The model identified a 7-day seasonality by comparing the last 3 Mondays, then applied a weighted average favoring the most recent cycle while adjusting for a 5% upward linear trend.'",
      "success_insight": "A concise, actionable 'Golden Rule' extracted from this success (e.g., 'When Covariate A leads Target B by 3 periods, treat the first sign of covariate reversal as a hard change-point for the forecast shape, regardless of the target's current trend.')",
      "meta_logic": {{
          "regime_identified": "stable|trending|post-shock recovery",
          "covariate_impact": "How the external variable modified the base trend",
          "inferred_shape": "mean reversion|continued drawdown|slow recovery"
      }},
      "tags": ["seasonality-detection", "trend-extrapolation", "outlier-handling"],
      "confidence": 0.8
    }}
  ]
}}
```
"""
