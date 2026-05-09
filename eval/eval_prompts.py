"""Prompts for semantic mismatch evaluation."""


def build_semantic_mismatch_prompt(question: str, gold_sql: str, pred_sql: str) -> str:
    return f"""
You are a strict SQL semantic mismatch classifier.
Return valid JSON only. Do not use markdown fences.
Output must start with "{{" and end with "}}".

Analyze semantic mismatch between GOLD_SQL and PREDICTED_SQL.

Question:
{question}

GOLD_SQL:
{gold_sql}

PREDICTED_SQL:
{pred_sql}

Output JSON schema:
{{
  "primary_error_type": "one short snake_case tag",
  "secondary_error_types": ["zero or more snake_case tags"],
  "confidence": 0.0,
  "reasoning": "concise explanation",
  "evidence": [
    "quote short SQL fragment from GOLD or PREDICTED"
  ]
}}

Common tags you may use:
- geo_pair_to_split_in
- geo_pair_to_bbox
- landcover_filter_dropped
- array_semantics_missing
- hallucinated_landcover_join
- output_projection_missing
- temporal_constraint_mismatch
- aggregation_target_mismatch
- over_broad_result_scope
- other_semantic_mismatch
""".strip()


def build_json_repair_prompt(content: str) -> str:
    return f"""
Convert the following text to the target JSON schema.
Return JSON only. Start with "{{" and end with "}}".

Text:
{content}

Target schema:
{{
  "primary_error_type": "one short snake_case tag",
  "secondary_error_types": ["zero or more snake_case tags"],
  "confidence": 0.0,
  "reasoning": "concise explanation",
  "evidence": ["short SQL fragment"]
}}
""".strip()
