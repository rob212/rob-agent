import os
from typing import List, Optional
from openai import AsyncOpenAI
from .base_llm import BaseLlm
from .llm_request import LlmRequest
from .llm_response import LlmResponse
from ..types.contents import Message, ToolCall, ContentItem
from pydantic import Field, PrivateAttr


class OpenAILlm(BaseLlm):
    """OpenAI LLM implementation."""
    
    _client: AsyncOpenAI = PrivateAttr()
    
    def __init__(self, model: str = "gpt-4o-mini", **kwargs):
        super().__init__(model=model, **kwargs)
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def generate(self, request: LlmRequest) -> LlmResponse:
        """Generate response from OpenAI."""
        messages = self._build_messages(request)
        tools = self._build_tools(request)
        
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice=request.tool_choice if tools else None,
            )
            
            return self._parse_response(response)
        except Exception as e:
            return LlmResponse(error_message=str(e))
    
    def _build_messages(self, request: LlmRequest) -> List[dict]:
        """Convert LlmRequest contents to OpenAI message format."""
        messages = []
        
        if request.instructions:
            messages.append({
                "role": "system",
                "content": "\n".join(request.instructions)
            })
        
        for item in request.contents:
            if isinstance(item, Message):
                messages.append({
                    "role": item.role,
                    "content": item.content
                })
            elif isinstance(item, ToolCall):
                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": item.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": item.name,
                            "arguments": str(item.arguments)
                        }
                    }]
                })
            elif hasattr(item, 'type') and item.type == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.tool_call_id,
                    "content": str(item.content)
                })
        
        return messages
    
    def _build_tools(self, request: LlmRequest) -> List[dict]:
        """Extract tool definitions from request."""
        if not request.tools_dict:
            return None
        
        return list(request.tools_dict.values())
    
    def _parse_response(self, response) -> LlmResponse:
        """Parse OpenAI response into LlmResponse."""
        choice = response.choices[0]
        message = choice.message
        content: List[ContentItem] = []
        
        if message.content:
            content.append(Message(
                role="assistant",
                content=message.content
            ))
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                import json
                content.append(ToolCall(
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments)
                ))
        
        return LlmResponse(
            content=content,
            usage_metadata={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        )
