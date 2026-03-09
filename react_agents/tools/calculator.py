from typing import Any
from pydantic import BaseModel, Field
from .base_tool import BaseTool
from ..models.execution_context import ExecutionContext


class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")


class Calculator(BaseTool):
    """Calculate mathematical expressions."""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Calculate mathematical expressions",
            pydantic_input_model=CalculatorInput
        )
    
    async def execute(self, context: ExecutionContext, expression: str) -> float:
        """Execute the calculation."""
        return eval(expression)


calculator = Calculator()
