from dataclasses import dataclass
from pydantic import BaseModel
from react_agents.models.execution_context import ExecutionContext

@dataclass
class AgentResult:
    """Result of an agent execution."""
    output: str | BaseModel
    context: ExecutionContext