"""Task dataclass adapted for the Meteo dataset.

Original BIRD fields like db_id and evidence are retained for pipeline
compatibility but assigned sensible defaults when not present.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List


@dataclass
class Task:
    """Represents a task (one evaluation question) in the pipeline.

    Attributes:
        question_id: Unique identifier for the question.
        db_id: Database identifier (always 'meteo' for our dataset).
        question: The full question text (+ evidence if any).
        evidence: Supporting evidence / hint (empty string if none).
        SQL: Ground-truth SQL query, if available.
        difficulty: Difficulty level ('A', 'B', 'C', …).
        raw_question: The original question text before evidence merging.
        category: Category label from the Meteo dataset.
    """
    question_id: int = field(init=False)
    db_id: str = field(init=False)
    question: str = field(init=False)
    evidence: str = field(init=False)
    SQL: Optional[str] = field(init=False, default=None)
    difficulty: Optional[str] = field(init=False, default=None)
    raw_question: str = field(init=False)
    question_toks: List[str] = field(default_factory=list)
    query: str = field(init=False, default="")
    category: Optional[str] = field(init=False, default=None)

    def __init__(self, task_data: Dict[str, Any]):
        self.question_id = task_data.get("question_id", 0)
        # In Meteo all questions share the same database
        self.db_id = task_data.get("db_id", "meteo")
        self.category = task_data.get("category") or task_data.get("Category")

        # Question text
        raw_q = (
            task_data.get("question")
            or task_data.get("Natural language question")
            or ""
        )
        self.raw_question = raw_q.strip()

        # Evidence / hint
        self.evidence = (task_data.get("evidence") or "").strip()
        if self.evidence:
            self.question = f"{self.raw_question} {self.evidence}".strip()
        else:
            self.question = self.raw_question
            self.evidence = "None"

        # Ground-truth SQL
        self.SQL = task_data.get("SQL") or task_data.get("ground_truth_sql") or task_data.get("SQL query")

        self.difficulty = task_data.get("difficulty") or self.category
        self.query = self.SQL or ""
