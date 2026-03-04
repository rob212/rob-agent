import inspect
import json
from litellm import completion


def function_to_input_schema(func) -> dict:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [param.name for param in signature.parameters.values()]

    return {
        "type": "object",
        "properties": parameters,
        "required": required,
    }


def format_tool_definition(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _tool_execution(tool_box, tool_call):
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    tool_result = tool_box[function_name](**function_args)
    return tool_result


def function_to_tool_definition(func) -> dict:
    return _format_tool_definition(
        func.__name__, func.__doc__ or "", _function_to_input_schema(func)
    )


def simple_agent_loop(system_prompt: str, question: str, tooling, model: str):
    tools = tooling
    tool_box = {tool.__name__: tool for tool in tools}
    tool_definitions = [function_to_tool_definition(tool) for tool in tools]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    while True:
        response = completion(model=model, messages=messages, tools=tool_definitions)

        assistant_message = response.choices[0].message

        if assistant_message.tool_calls:
            messages.append(assistant_message)
            for tool_call in assistant_message.tool_calls:
                tool_result = _tool_execution(tool_box, tool_call)
                messages.append(
                    {
                        "role": "tool",
                        "content": str(tool_result),
                        "tool_call_id": tool_call.id,
                    }
                )
        else:
            return assistant_message.content, messages

def mcp_tools_to_openai_format(mcp_tools) -> list[dict]:
    """Convert MCP tool definitions to OpenAI tool format."""
    return [
        _format_tool_definition(
            name=tool.name,
            description=tool.description,
            parameters=tool.inputSchema,
        )
        for tool in mcp_tools.tools
    ]
 