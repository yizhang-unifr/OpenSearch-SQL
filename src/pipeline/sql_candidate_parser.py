import re
from typing import List, Optional

from pydantic import BaseModel

from runner.check_and_correct import sql_raw_parse


class SQLCandidate(BaseModel):
    sql: str
    reasoning: Optional[str] = None
    raw: str


def _extract_sql_from_text(text: str) -> str:
    raw_sql = sql_raw_parse(text, False)[0]
    if raw_sql:
        return raw_sql

    block_matches = re.findall(r"```sql(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in block_matches:
        candidate = block.strip()
        if re.search(r"\bSELECT\b", candidate, flags=re.IGNORECASE):
            return re.sub(r"\s+", " ", candidate).strip()

    select_match = re.search(r"(SELECT\b.*?;)", text, flags=re.IGNORECASE | re.DOTALL)
    if select_match:
        return re.sub(r"\s+", " ", select_match.group(1)).strip()

    select_match = re.search(r"(SELECT\b.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if select_match:
        return re.sub(r"\s+", " ", select_match.group(1)).strip()

    return ""


def _extract_reasoning(text: str) -> Optional[str]:
    if "#SQL:" in text:
        prefix = text.split("#SQL:", 1)[0].strip()
        return prefix or None
    if "```sql" in text.lower():
        prefix = re.split(r"```sql", text, flags=re.IGNORECASE, maxsplit=1)[0].strip()
        return prefix or None
    return None


def parse_sql_candidates(raw_candidates: List[str]) -> List[SQLCandidate]:
    parsed: List[SQLCandidate] = []
    for raw in raw_candidates:
        text = (raw or "").strip()
        sql = _extract_sql_from_text(text)
        reasoning = _extract_reasoning(text)
        parsed.append(SQLCandidate(sql=sql, reasoning=reasoning, raw=text))
    return parsed
