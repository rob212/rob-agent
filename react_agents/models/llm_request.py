from ..types.contents import ContentItem
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class LlmRequest(BaseModel):
    """Request object for LLM calls."""
    instructions: List[str] = Field(default_factory=list)
    contents: List[ContentItem] = Field(default_factory=list)
    tools_dict: Dict[str, Any] = Field(default_factory=dict)
    tool_choice: Optional[str] = None