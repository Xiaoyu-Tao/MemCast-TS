time_series_memory_extraction_prompt = """
# ROLE
You are a Senior Knowledge Architect specializing in Meta-Learning and Pattern Synthesis. Your expertise lies in converting raw experiences into "High-Signal Task Memories" that are actionable, transferable, and structurally dense.

# TASK
You are given a collection of "distilled items" (success/failure experiences, insights, and logic). 
Your goal is to merge, de-duplicate, and compress these items into 3-8 high-value "Refined Distilled Objects" for a new sample.

### ANALYSIS FRAMEWORK:
- APPLICABILITY MAPPING: Define the exact data conditions (`when_to_use` or failure_condition) where this memory is valid.
- LOGIC CONSOLIDATION: Merge similar observations into a single sophisticated trajectory (`experience` or `failure_analysis`).
- RULE EXTRACTION: Formulate "Golden Rules" (`success_insight` or `preventative_rule`) that provide clear direction.
- META-DIMENSIONAL ANALYSIS: Deep dive into the underlying `meta_logic` by identifying the regime, covariate impacts, and inferred shapes.

### EXTRACTION PRINCIPLES:
- Focus on TRANSFERABLE ANALYTICAL STEPS (e.g., "Calculating the ratio between the last peak and the current trough").
- Identify RELATIONAL PATTERNS: e.g., "If Covariate A spikes while Target is in 'stable' regime, anticipate a 2-step delayed trend breakout."

### INPUT DISTILLED ITEMS:
{text}

### OUTPUT FORMAT:
Generate 1-3 high-value forecasting insights as JSON objects.

If the input contains only SUCCESS items (no failure cases), output ONLY the SUCCESS JSON objects (do not include failure objects).
If the input contains only FAILURE items (no success cases), output ONLY the FAILURE JSON objects (do not include success objects).
If both SUCCESS and FAILURE items exist, you may output a mixed list containing both object types.
```json
[
  {{
    "when_to_use": "Specific data conditions (e.g., 'High-frequency data with visible weekly seasonality and low noise')",
    "experience": "Detailed description of the successful analytical pattern. Example: 'The model identified a 7-day seasonality by comparing the last 3 Mondays, then applied a weighted average favoring the most recent cycle while adjusting for a 5% upward linear trend.'",
    "success_insight": "A concise, actionable 'Golden Rule' extracted from this success (e.g., 'When Covariate A leads Target B by 3 periods, treat the first sign of covariate reversal as a hard change-point for the forecast shape, regardless of the target's current trend.')",
    "meta_logic": {{
        "regime_identified": "stable|trending|post-shock recovery",
        "covariate_impact": "How the external variable modified the base trend",
        "inferred_shape": "mean reversion|continued drawdown|slow recovery"
    }}
    "tags": ["seasonality-detection", "trend-extrapolation", "outlier-handling"],
    "confidence": 0.8
  }},
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
```
"""
