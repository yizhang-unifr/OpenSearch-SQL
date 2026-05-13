import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Tuple

@dataclass
class Statistics:
    corrects: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    incorrects: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    errors: Dict[str, List[Union[Tuple[str, str], Tuple[str, str, str]]]] = field(default_factory=dict)
    total: Dict[str, int] = field(default_factory=dict)
    ves_sum: Dict[str, float] = field(default_factory=dict)

    def _ex(self, key: str) -> float:
        n = self.total.get(key, 0)
        return len(self.corrects.get(key, [])) / n if n else 0.0

    def _ves(self, key: str) -> float:
        n = self.total.get(key, 0)
        return self.ves_sum.get(key, 0.0) / n if n else 0.0

    def to_dict(self) -> Dict[str, Dict[str, Union[Dict[str, int], List[Tuple[str, str]]]]]:
        return {
            "counts": {
                key: {
                    "correct": len(self.corrects.get(key, [])),
                    "incorrect": len(self.incorrects.get(key, [])),
                    "error": len(self.errors.get(key, [])),
                    "total": self.total.get(key, 0),
                    "EX": round(self._ex(key), 4),
                    "VES": round(self._ves(key), 4),
                }
                for key in self.total
            },
            "ids": {
                key: {
                    "correct": sorted(self.corrects.get(key, [])),
                    "incorrect": sorted(self.incorrects.get(key, [])),
                    "error": sorted(self.errors.get(key, []))
                }
                for key in self.total
            }
        }


class StatisticsManager:
    def __init__(self, result_directory: str):
        """
        Initializes the StatisticsManager.

        Args:
            result_directory (str): The directory to store results.
        """
        self.result_directory = Path(result_directory)
        self.statistics = Statistics()

        # Ensure the statistics file exists
        self.statistics_file_path = self.result_directory / "-statistics.json"
        if not self.statistics_file_path.exists():
            self.statistics_file_path.touch()
            self.dump_statistics_to_file()

    def update_stats(self, db_id: str, question_id: str, evaluation_for: str, result: Dict[str, Any]):
        exec_res = result["exec_res"]
        exec_err = result["exec_err"]
        ves = float(result.get("ves", 0.0))

        self.statistics.total[evaluation_for] = self.statistics.total.get(evaluation_for, 0) + 1
        self.statistics.ves_sum[evaluation_for] = (
            self.statistics.ves_sum.get(evaluation_for, 0.0) + ves
        )

        if exec_res == 1:
            if evaluation_for not in self.statistics.corrects:
                self.statistics.corrects[evaluation_for] = []
            self.statistics.corrects[evaluation_for].append((db_id, question_id))
        else:
            if exec_err == "incorrect answer":
                if evaluation_for not in self.statistics.incorrects:
                    self.statistics.incorrects[evaluation_for] = []
                self.statistics.incorrects[evaluation_for].append((db_id, question_id))
            else:
                if evaluation_for not in self.statistics.errors:
                    self.statistics.errors[evaluation_for] = []
                self.statistics.errors[evaluation_for].append((db_id, question_id, exec_err))

    def dump_statistics_to_file(self):
        """
        Dumps the current statistics to a JSON file.
        """
        with self.statistics_file_path.open('w') as f:
            json.dump(self.statistics.to_dict(), f, indent=4,ensure_ascii=False)
