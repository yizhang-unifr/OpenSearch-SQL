from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


@dataclass
class PluginContext:
    question: str
    geo_context: Dict[str, Any]
    ontology_grounded_function: Dict[str, Any]
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginResult:
    sql: str
    changed: bool = False
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SQLPlugin(Protocol):
    def name(self) -> str:
        ...

    def applies(self, context: PluginContext) -> bool:
        ...

    def transform(self, sql: str, context: PluginContext) -> PluginResult:
        ...
