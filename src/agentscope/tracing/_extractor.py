# -*- coding: utf-8 -*-
"""Utils for tracing."""
import inspect
from typing import Any, Dict, Tuple, TYPE_CHECKING

from .. import _config
from ..embedding._embedding_base import EmbeddingModelBase
from ..model._model_base import ChatModelBase
from ._attributes import (
    SpanAttributes,
    OperationNameValues,
    ProviderNameValues,
)
from ._utils import _serialize_to_str
from ._converter import (
    _convert_block_to_part,
)

if TYPE_CHECKING:
    from ..agent import AgentBase
    from ..formatter import FormatterBase
    from ..tool import (
        Toolkit,
        ToolResponse,
    )
    from ..message import (
        Msg,
        ToolUseBlock,
    )
    from ..embedding import EmbeddingResponse
    from ..model import ChatResponse
    from opentelemetry.trace import Span
else:
    Toolkit = "Toolkit"
    ToolResponse = "ToolResponse"
    Msg = "Msg"
    ToolUseBlock = "ToolUseBlock"
    EmbeddingResponse = "EmbeddingResponse"
    ChatResponse = "ChatResponse"
    Span = "Span"


_FORMATTER_MAP = {
    "DashScopeChatFormatter": ProviderNameValues.DASHSCOPE,
    "DashScopeMultiAgentFormatter": ProviderNameValues.DASHSCOPE,
    "OpenAIChatFormatter": ProviderNameValues.OPENAI,
    "OpenAIMultiAgentFormatter": ProviderNameValues.OPENAI,
    "AnthropicChatFormatter": ProviderNameValues.ANTHROPIC,
    "AnthropicMultiAgentFormatter": ProviderNameValues.ANTHROPIC,
    "GeminiChatFormatter": ProviderNameValues.GCP_GEMINI,
    "GeminiMultiAgentFormatter": ProviderNameValues.GCP_GEMINI,
    "OllamaChatFormatter": ProviderNameValues.OLLAMA,
    "OllamaMultiAgentFormatter": ProviderNameValues.OLLAMA,
    "DeepSeekChatFormatter": ProviderNameValues.DEEPSEEK,
    "DeepSeekMultiAgentFormatter": ProviderNameValues.DEEPSEEK,
}

# Map model class names to provider names
_MODEL_PROVIDER_MAP = {
    "DashScopeChatModel": ProviderNameValues.DASHSCOPE,
    "OpenAIChatModel": ProviderNameValues.OPENAI,
    "AnthropicChatModel": ProviderNameValues.ANTHROPIC,
    "GeminiChatModel": ProviderNameValues.GCP_GEMINI,
    "OllamaChatModel": ProviderNameValues.OLLAMA,
    "TrinityChatModel": ProviderNameValues.OPENAI,
}

# Map base URL fragments to provider names for OpenAI-compatible APIs
_BASE_URL_PROVIDER_MAP = [
    ("api.openai.com", ProviderNameValues.OPENAI),
    ("dashscope", ProviderNameValues.DASHSCOPE),
    ("deepseek", ProviderNameValues.DEEPSEEK),
    ("moonshot", ProviderNameValues.MOONSHOT),
    ("generativelanguage.googleapis.com", ProviderNameValues.GCP_GEMINI),
    ("openai.azure.com", ProviderNameValues.AZURE_AI_OPENAI),
    ("amazonaws.com", ProviderNameValues.AWS_BEDROCK),
]


def get_common_attributes() -> Dict[str, str]:
    """Get common attributes for all spans.

    Returns:
        Dict[str, str]: Common span attributes including conversation ID
    """
    return {
        SpanAttributes.GEN_AI_CONVERSATION_ID: _serialize_to_str(
            _config.run_id,
        ),
    }


def _get_format_target(instance: Any) -> str:
    """Get format target for the given instance.

    Maps AgentScope class names to format target names.

    Args:
        instance: Formatter instance

    Returns:
        str: Format target name or "unknown"
    """
    classname = instance.__class__.__name__
    return _FORMATTER_MAP.get(classname, "unknown")


def _get_provider_name(instance: ChatModelBase) -> str:
    """Get provider name from ChatModelBase instance.

    Maps ChatModelBase class names to provider names, with special handling
    for OpenAI-compatible APIs that may use different base URLs.
    This follows the implementation pattern from agentscope-java PR #73.

    Args:
        instance: ChatModelBase instance

    Returns:
        str: Provider name (e.g., "openai", "dashscope", "anthropic")
    """
    classname = instance.__class__.__name__

    # Special handling for OpenAIChatModel - check base_url
    if classname == "OpenAIChatModel":
        # Try to get base_url from the client
        base_url = None
        if hasattr(instance, "client") and hasattr(
            instance.client,
            "base_url",
        ):
            base_url = str(instance.client.base_url)

        # If base_url is None or empty, return default OpenAI
        if not base_url:
            return ProviderNameValues.OPENAI

        # Check base_url fragments to identify provider
        for url_fragment, provider_name in _BASE_URL_PROVIDER_MAP:
            if url_fragment in base_url:
                return provider_name

        # If no match found, return openai as default
        return ProviderNameValues.OPENAI

    # For other model types, use direct mapping
    return _MODEL_PROVIDER_MAP.get(classname, "unknown")


def _get_tool_definitions(
    tools: list[dict[str, Any]] | None,
    tool_choice: str | None,
) -> str | None:
    """Extract and serialize tool definitions for tracing.

    Converts AgentScope/OpenAI nested tool format to OpenTelemetry GenAI
    flat format for tracing.

    Args:
        tools: List of tool definitions in OpenAI format with nested structure:
            [{"type": "function", "function": {"name": ...,
             "parameters": ...}}]
        tool_choice: Tool choice mode (auto, none, any, required, or tool name)

    Returns:
        str | None: Serialized tool definitions in flat format:
            [{"type": "function", "name": ..., "parameters": ...}]
            or None if tools should not be traced
    """
    #  No tools provided
    if tools is None or not isinstance(tools, list) or len(tools) == 0:
        return None

    #  Tool choice is explicitly "none" (model should not use tools)
    if tool_choice == "none":
        return None

    try:
        # Convert nested format to flat format for OpenTelemetry GenAI
        # TODO: Currently only supports "function" type tools. If other tool
        # types are added in the future (e.g., "retrieval", "code_interpreter",
        # "browser"), this conversion logic needs to be updated to handle them.
        flat_tools = []
        for tool in tools:
            if not isinstance(tool, dict) or "function" not in tool:
                continue

            func_def = tool["function"]
            flat_tool = {
                "type": tool.get("type", "function"),
                "name": func_def.get("name"),
                "description": func_def.get("description"),
                "parameters": func_def.get("parameters"),
            }
            # Remove None values
            flat_tool = {k: v for k, v in flat_tool.items() if v is not None}
            flat_tools.append(flat_tool)

        if flat_tools:
            return _serialize_to_str(flat_tools)
        return None

    except Exception:
        return None


def get_llm_request_attributes(
    instance: ChatModelBase,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Dict[str, str]:
    """Get LLM request attributes for OpenTelemetry tracing.

    Extracts request parameters from LLM model calls into GenAI attributes.

    Args:
        instance: ChatModelBase instance making the request
        args: Positional arguments
        kwargs: Keyword arguments including generation parameters

    Returns:
        Dict[str, str]: OpenTelemetry GenAI attributes with string values
    """

    attributes = {
        # required attributes
        SpanAttributes.GEN_AI_OPERATION_NAME: OperationNameValues.CHAT,
        SpanAttributes.GEN_AI_PROVIDER_NAME: _get_provider_name(instance),
        # conditionally required attributes
        SpanAttributes.GEN_AI_REQUEST_MODEL: getattr(
            instance,
            "model_name",
            "unknown_model",
        ),
        # recommended attributes
        SpanAttributes.GEN_AI_REQUEST_TEMPERATURE: kwargs.get("temperature"),
        SpanAttributes.GEN_AI_REQUEST_TOP_P: kwargs.get("p")
        or kwargs.get("top_p"),
        SpanAttributes.GEN_AI_REQUEST_TOP_K: kwargs.get("top_k"),
        SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS: kwargs.get("max_tokens"),
        SpanAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY: kwargs.get(
            "presence_penalty",
        ),
        SpanAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY: kwargs.get(
            "frequency_penalty",
        ),
        SpanAttributes.GEN_AI_REQUEST_STOP_SEQUENCES: kwargs.get(
            "stop_sequences",
        ),
        SpanAttributes.GEN_AI_REQUEST_SEED: kwargs.get("seed"),
        # custom attributes
        SpanAttributes.AGENTSCOPE_FUNCTION_INPUT: _serialize_to_str(
            {
                "args": args,
                "kwargs": kwargs,
            },
        ),
    }

    # Extract tool definitions if provided
    tool_definitions = _get_tool_definitions(
        tools=kwargs.get("tools"),
        tool_choice=kwargs.get("tool_choice"),
    )
    if tool_definitions:
        attributes[SpanAttributes.GEN_AI_TOOL_DEFINITIONS] = tool_definitions

    return {k: v for k, v in attributes.items() if v is not None}


def get_llm_span_name(attributes: Dict[str, str]) -> str:
    """Generate span name for LLM operations.

    Args:
        attributes: LLM request attributes dict

    Returns:
        str: Formatted span name "{operation} {model}"
    """
    return (
        f"{attributes[SpanAttributes.GEN_AI_OPERATION_NAME]} "
        f"{attributes[SpanAttributes.GEN_AI_REQUEST_MODEL]}"
    )


def _get_llm_output_messages(
    chat_response: Any,
) -> list[dict[str, Any]]:
    """Extract and format LLM output messages for tracing.

    Converts ChatResponse objects to standardized message format.

    Args:
        chat_response: Chat response object with content blocks

    Returns:
        list[dict[str, Any]]: List containing formatted message
    """
    try:
        from agentscope.model import ChatResponse

        if not isinstance(chat_response, ChatResponse):
            return chat_response

        parts = []
        finish_reason = "stop"  # 默认完成原因

        for block in chat_response.content:
            part = _convert_block_to_part(block)
            if part:
                parts.append(part)

        output_message = {
            "role": "assistant",
            "parts": parts,
            "finish_reason": finish_reason,
        }

        return [output_message]

    except Exception:
        return [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "text",
                        "content": "<error processing response>",
                    },
                ],
                "finish_reason": "error",
            },
        ]


def get_llm_response_attributes(
    chat_response: Any,
) -> Dict[str, str]:
    """Get LLM response attributes for OpenTelemetry tracing.

    Extracts response metadata and formats into GenAI attributes.

    Args:
        chat_response: Chat response object with data and usage info

    Returns:
        Dict[str, str]: OpenTelemetry GenAI response attributes
    """
    attributes = {
        SpanAttributes.GEN_AI_RESPONSE_ID: getattr(
            chat_response,
            "id",
            "unknown_id",
        ),
        # FIXME: finish reason should be capture in chat response
        SpanAttributes.GEN_AI_RESPONSE_FINISH_REASONS: '["stop"]',
    }
    if hasattr(chat_response, "usage") and chat_response.usage:
        attributes[
            SpanAttributes.GEN_AI_USAGE_INPUT_TOKENS
        ] = chat_response.usage.input_tokens
        attributes[
            SpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
        ] = chat_response.usage.output_tokens

    output_messages = _get_llm_output_messages(chat_response)
    if output_messages:
        attributes[SpanAttributes.GEN_AI_OUTPUT_MESSAGES] = _serialize_to_str(
            output_messages,
        )

    attributes[SpanAttributes.AGENTSCOPE_FUNCTION_OUTPUT] = _serialize_to_str(
        chat_response,
    )
    return attributes


def _get_agent_messages(
    msg: Msg,
) -> dict[str, Any]:
    """Convert AgentScope message to standardized parts format.

    Transforms Msg objects into OpenTelemetry GenAI format.

    Args:
        msg: AgentScope message object with content blocks

    Returns:
        dict[str, Any]: Formatted message dictionary
    """
    try:
        parts = []
        # 遍历所有内容块
        for block in msg.get_content_blocks():
            part = _convert_block_to_part(block)
            if part:
                parts.append(part)

        formatted_msg = {
            "role": msg.role,
            "parts": parts,
            "name": msg.name,
            "finish_reason": "stop",
        }

        if msg.name:
            formatted_msg["name"] = msg.name

        return formatted_msg

    except Exception:
        return {
            "role": msg.role,
            "parts": [
                {
                    "type": "text",
                    "content": str(msg.content) if msg.content else "",
                },
            ],
            "name": msg.name,
            "finish_reason": "stop",
        }


def get_agent_request_attributes(
    instance: "AgentBase",
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Dict[str, str]:
    """Get agent request attributes for OpenTelemetry tracing.

    Extracts agent metadata and input data into GenAI attributes.

    Args:
        instance: AgentBase instance being invoked
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Dict[str, str]: OpenTelemetry GenAI agent attributes
    """
    attributes = {
        SpanAttributes.GEN_AI_OPERATION_NAME: (
            OperationNameValues.INVOKE_AGENT
        ),
        SpanAttributes.GEN_AI_AGENT_ID: getattr(instance, "id", "unknown"),
        SpanAttributes.GEN_AI_AGENT_NAME: getattr(
            instance,
            "name",
            "unknown_agent",
        ),
        SpanAttributes.GEN_AI_AGENT_DESCRIPTION: inspect.getdoc(
            instance.__class__,
        )
        or "No description available",
    }

    msg = None
    if args and len(args) > 0:
        msg = args[0]
    elif "msg" in kwargs:
        msg = kwargs["msg"]
    if msg:
        input_messages = _get_agent_messages(msg)
        attributes[SpanAttributes.GEN_AI_INPUT_MESSAGES] = _serialize_to_str(
            input_messages,
        )

    # custom attributes
    attributes[SpanAttributes.AGENTSCOPE_FUNCTION_INPUT] = _serialize_to_str(
        {
            "args": args,
            "kwargs": kwargs,
        },
    )
    return attributes


def get_agent_span_name(attributes: Dict[str, str]) -> str:
    """Generate span name for agent operations.

    Args:
        attributes: Agent request attributes dict

    Returns:
        str: Formatted span name "{operation} {agent_name}"
    """
    return (
        f"{attributes[SpanAttributes.GEN_AI_OPERATION_NAME]} "
        f"{attributes[SpanAttributes.GEN_AI_AGENT_NAME]}"
    )


def get_agent_response_attributes(
    agent_response: Any,
) -> Dict[str, str]:
    """Get agent response attributes for OpenTelemetry tracing.

    Args:
        agent_response: Response object returned by agent

    Returns:
        Dict[str, str]: OpenTelemetry GenAI response attributes
    """
    attributes = {
        SpanAttributes.GEN_AI_OUTPUT_MESSAGES: _serialize_to_str(
            _get_agent_messages(agent_response),
        ),
        SpanAttributes.AGENTSCOPE_FUNCTION_OUTPUT: _serialize_to_str(
            agent_response,
        ),
    }
    return attributes


def get_tool_request_attributes(
    instance: "Toolkit",
    tool_call: ToolUseBlock,
) -> Dict[str, str]:
    """Get tool request attributes for OpenTelemetry tracing.

    Extracts tool execution metadata into GenAI attributes.

    Args:
        instance: Toolkit instance with tool definitions
        tool_call: Tool use block with call information

    Returns:
        Dict[str, str]: OpenTelemetry GenAI tool attributes
    """
    attributes = {
        SpanAttributes.GEN_AI_OPERATION_NAME: (
            OperationNameValues.EXECUTE_TOOL
        ),
    }

    if tool_call:
        tool_name = tool_call.get("name")
        attributes[SpanAttributes.GEN_AI_TOOL_CALL_ID] = tool_call.get("id")
        attributes[SpanAttributes.GEN_AI_TOOL_NAME] = tool_name
        attributes[
            SpanAttributes.GEN_AI_TOOL_CALL_ARGUMENTS
        ] = _serialize_to_str(tool_call.get("input"))

        if tool_name:
            if tool := getattr(instance, "tools", {}).get(tool_name):
                if tool_func := getattr(tool, "json_schema", {}).get(
                    "function",
                    {},
                ):
                    attributes[
                        SpanAttributes.GEN_AI_TOOL_DESCRIPTION
                    ] = tool_func.get("description", "unknown_description")

        # custom attributes
        attributes[
            SpanAttributes.AGENTSCOPE_FUNCTION_INPUT
        ] = _serialize_to_str(
            {
                "tool_call": tool_call,
            },
        )
    return attributes


def get_tool_span_name(attributes: Dict[str, str]) -> str:
    """Generate span name for tool operations.

    Args:
        attributes: Tool request attributes dict

    Returns:
        str: Formatted span name "{operation} {tool_name}"
    """
    return (
        f"{attributes[SpanAttributes.GEN_AI_OPERATION_NAME]} "
        f"{attributes[SpanAttributes.GEN_AI_TOOL_NAME]}"
    )


def get_tool_response_attributes(
    tool_response: Any,
) -> Dict[str, str]:
    """Get tool response attributes for OpenTelemetry tracing.

    Args:
        tool_response: Response object from tool execution

    Returns:
        Dict[str, str]: OpenTelemetry GenAI response attributes
    """
    attributes = {
        SpanAttributes.GEN_AI_TOOL_CALL_RESULT: _serialize_to_str(
            tool_response,
        ),
    }

    attributes[SpanAttributes.AGENTSCOPE_FUNCTION_OUTPUT] = _serialize_to_str(
        tool_response,
    )
    return attributes


def get_formatter_request_attributes(
    instance: "FormatterBase",
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Dict[str, str]:
    """Get formatter request attributes for OpenTelemetry tracing.

    Extracts formatter metadata into GenAI attributes.

    Args:
        instance: FormatterBase instance being used
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Dict[str, str]: OpenTelemetry GenAI formatter attributes
    """
    attributes = {
        SpanAttributes.GEN_AI_OPERATION_NAME: (OperationNameValues.FORMATTER),
        SpanAttributes.AGENTSCOPE_FORMAT_TARGET: _get_format_target(instance),
        SpanAttributes.AGENTSCOPE_FUNCTION_INPUT: _serialize_to_str(
            {
                "args": args,
                "kwargs": kwargs,
            },
        ),
    }
    return attributes


def get_formatter_span_name(attributes: Dict[str, str]) -> str:
    """Generate span name for formatter operations.

    Args:
        attributes: Formatter request attributes dict

    Returns:
        str: Formatted span name "{operation} {provider}"
    """
    return (
        f"{attributes[SpanAttributes.GEN_AI_OPERATION_NAME]} "
        f"{attributes[SpanAttributes.AGENTSCOPE_FORMAT_TARGET]}"
    )


def get_formatter_response_attributes(
    response: Any,
) -> Dict[str, str]:
    """Get formatter response attributes for OpenTelemetry tracing.

    Args:
        response: Response object from formatter (list[dict])

    Returns:
        Dict[str, str]: OpenTelemetry GenAI response attributes
    """
    attributes = {
        SpanAttributes.AGENTSCOPE_FUNCTION_OUTPUT: _serialize_to_str(response),
    }
    if isinstance(response, list):
        attributes[SpanAttributes.AGENTSCOPE_FORMAT_COUNT] = len(response)

    return attributes


def get_generic_function_request_attributes(
    function_name: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Dict[str, str]:
    """Get generic function request attributes for tracing.

    Extracts metadata from function calls into GenAI attributes.

    Args:
        function_name: Name of function being called
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Dict[str, str]: OpenTelemetry GenAI function attributes
    """
    attributes = {
        SpanAttributes.GEN_AI_OPERATION_NAME: (
            OperationNameValues.INVOKE_GENERIC_FUNCTION
        ),
        SpanAttributes.AGENTSCOPE_FUNCTION_NAME: function_name,
        SpanAttributes.AGENTSCOPE_FUNCTION_INPUT: _serialize_to_str(
            {
                "args": args,
                "kwargs": kwargs,
            },
        ),
    }
    return attributes


def get_generic_function_span_name(attributes: Dict[str, str]) -> str:
    """Generate span name for generic function operations.

    Args:
        attributes: Generic function request attributes dict

    Returns:
        str: Formatted span name "{operation} {function_name}"
    """
    return (
        f"{attributes[SpanAttributes.GEN_AI_OPERATION_NAME]} "
        f"{attributes[SpanAttributes.AGENTSCOPE_FUNCTION_NAME]}"
    )


def get_generic_function_response_attributes(
    response: Any,
) -> Dict[str, str]:
    """Get generic function response attributes for tracing.

    Args:
        response: Response object from generic function

    Returns:
        Dict[str, str]: OpenTelemetry GenAI response attributes
    """
    attributes = {
        SpanAttributes.AGENTSCOPE_FUNCTION_OUTPUT: _serialize_to_str(response),
    }
    return attributes


def get_embedding_request_attributes(
    instance: "EmbeddingModelBase",
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Dict[str, str]:
    """Get embedding request attributes for OpenTelemetry tracing.

    Extracts embedding model metadata into GenAI attributes.

    Args:
        instance: EmbeddingModelBase instance making request
        args: Positional arguments
        kwargs: Keyword arguments including dimensions

    Returns:
        Dict[str, str]: OpenTelemetry GenAI attributes
    """
    attributes = {
        SpanAttributes.GEN_AI_OPERATION_NAME: OperationNameValues.EMBEDDINGS,
        SpanAttributes.GEN_AI_REQUEST_MODEL: getattr(
            instance,
            "model_name",
            "unknown_model",
        ),
        SpanAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT: kwargs.get(
            "dimensions",
        ),
        SpanAttributes.AGENTSCOPE_FUNCTION_INPUT: _serialize_to_str(
            {
                "args": args,
                "kwargs": kwargs,
            },
        ),
    }
    return attributes


def get_embedding_span_name(attributes: Dict[str, str]) -> str:
    """Generate span name for embedding operations.

    Args:
        attributes: Embedding request attributes dict

    Returns:
        str: Formatted span name "{operation} {model}"
    """
    return (
        f"{attributes[SpanAttributes.GEN_AI_OPERATION_NAME]} "
        f"{attributes[SpanAttributes.GEN_AI_REQUEST_MODEL]}"
    )


def get_embedding_response_attributes(
    response: Any,
) -> Dict[str, str]:
    """Get embedding response attributes for OpenTelemetry tracing.

    Args:
        response: Response object from embedding model

    Returns:
        Dict[str, str]: OpenTelemetry GenAI response attributes
    """
    attributes = {
        SpanAttributes.AGENTSCOPE_FUNCTION_OUTPUT: _serialize_to_str(response),
    }
    return attributes
