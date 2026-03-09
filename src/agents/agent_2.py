import asyncio
from dotenv import load_dotenv

from typing import List, Optional
from react_agents.models import BaseLlm
from react_agents.models import LlmRequest
from react_agents.models import LlmResponse
from react_agents.models import AgentResult
from react_agents.types.contents import Message, ToolCall
from react_agents.types import Event
from react_agents.models import ExecutionContext
from react_agents.tools import BaseTool
from react_agents.types.contents import ToolResult
from typing import Type
from pydantic import BaseModel

class Agent:
    def __init__(self, name: str, model: BaseLlm, tools: List[BaseTool], instructions: str, max_steps: int = 10, output_type: Optional[Type[BaseModel]] = None):
        self.name = name
        self.model = model
        self.max_steps = max_steps
        self.instructions = instructions
        self.tools = self._setup_tools(tools)
        
    def _setup_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        return tools
    
    async def run(
        self,
        user_input: str,
        context: ExecutionContext = None
    ) -> str:
        # Create or reuse context
        if context is None:
            context = ExecutionContext()

        # Add user input as the first event
        user_event = Event(
            execution_id=context.execution_id,
            author="user",
            content=[Message(role="user", content=user_input)]
        )
        context.add_event(user_event)

        # Execute steps until completion or max steps reached
        while not context.final_result and context.current_step < self.max_steps:
            await self.step(context)

            # Check if the last event is a final response
            last_event = context.events[-1]
            if self._is_final_response(last_event):
                context.final_result = self._extract_final_result(last_event)

        return AgentResult(output=context.final_result, context=context)
    
    async def step(self, context: ExecutionContext):
        # for visibility as we experiment and learn
        print(f"[Step {context.current_step + 1}]")
        # Prepare what to send to the LLM
        llm_request = self._prepare_llm_request(context)

        # Get LLM's decision
        llm_response = await self.think(llm_request)

        # Record LLM response as an event
        response_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            content=llm_response.content,
        )
        context.add_event(response_event)

        # Execute tools if the LLM requested any
        tool_calls = [c for c in llm_response.content if isinstance(c, ToolCall)]
        if tool_calls:
            tool_results = await self.act(context, tool_calls)
            tool_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                content=tool_results,
            )
            context.add_event(tool_event)

        context.increment_step()
        
    async def think(self, llm_request: LlmRequest) -> LlmResponse:
        return await self.model.generate(llm_request)
    
    async def act(
        self, 
        context: ExecutionContext, 
        tool_calls: List[ToolCall]
    ) -> List[ToolResult]:
        tools_dict = {tool.name: tool for tool in self.tools}
        results = []
        
        for tool_call in tool_calls:
            if tool_call.name not in tools_dict:
                raise ValueError(f"Tool '{tool_call.name}' not found")
            
            tool = tools_dict[tool_call.name]
            
            try:
                output = await tool(context, **tool_call.arguments)
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="success",
                    content=[output],
                ))
            except Exception as e:
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="error",
                    content=[str(e)],
                ))
        
        return results
    
    def _is_final_response(self, event: Event) -> bool:
        """Check if this event contains a final response."""
        has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
        has_tool_results = any(isinstance(c, ToolResult) for c in event.content)
        return not has_tool_calls and not has_tool_results

    def _extract_final_result(self, event: Event) -> str:
        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content
        return None
                
    def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
        # Flatten events into content items
        flat_contents = []
        for event in context.events:
            flat_contents.extend(event.content)

        return LlmRequest(
            instructions=[self.instructions] if self.instructions else [],
            contents=flat_contents,
            tools_dict= {tool.name: tool.tool_definition for tool in self.tools},
            tool_choice="auto" if self.tools else None,
        )
    
async def main():
    load_dotenv()
    
    # Import your tools and model
    from react_agents.tools import calculator
    from react_agents.models.openai import OpenAILlm  # or whatever LLM you're using
    
    # Create agent
    model = OpenAILlm(model="gpt-4o-mini")
    agent = Agent(
        name="TestAgent",
        model=model,
        tools=[calculator],
        instructions="You are a helpful assistant"
    )
    
    # Run and print results
    result = await agent.run("What is 1234 * 5678?")
    print(result.output)
    print(result.context.current_step)

if __name__ == "__main__":
    asyncio.run(main())