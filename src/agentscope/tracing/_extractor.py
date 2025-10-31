# -*- coding: utf-8 -*-
"""Utils for tracing."""
import inspect
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from .. import _config
from ..embedding._embedding_base import EmbeddingModelBase
from ..model._model_base import ChatModelBase
from ._attributes import SpanAttributes, OperationNameValues, ProviderNameValues, OldSpanKind
from ._utils import _serialize_to_str

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
        ContentBlock,
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
    ContentBlock = "ContentBlock"


_PROVIDER_MAP = {
    # LLM Models
    "OpenAIChatModel": ProviderNameValues.OPENAI,
    "GeminiChatModel": ProviderNameValues.GCP_GEMINI,
    "AnthropicChatModel": ProviderNameValues.ANTHROPIC,
    "DashScopeChatModel": ProviderNameValues.DASHSCOPE,
    "OllamaChatModel": ProviderNameValues.OLLAMA,
    # Formatter
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
    # Embedding Models
    "DashScopeTextEmbedding": ProviderNameValues.DASHSCOPE,
    "DashScopeMultiModalEmbedding": ProviderNameValues.DASHSCOPE,
    "OpenAITextEmbedding": ProviderNameValues.OPENAI,
    "GeminiTextEmbedding": ProviderNameValues.GCP_GEMINI,
    "OllamaTextEmbedding": ProviderNameValues.OLLAMA,
}


# pylint: disable=R0912
# pylint: disable=R0915
def _convert_block_to_part(block: ContentBlock) -> Optional[dict[str, Any]]:
    """Convert content block to OpenTelemetry GenAI part format.

    Converts text, thinking, tool_use, tool_result, image, audio, video
    blocks to standardized parts.

    Args:
        block: ContentBlock object

    Returns:
        Optional[dict[str, Any]]: Standardized part dict or None
    """
    block_type = block.get("type")

    if block_type == "text":
        part = {
            "type": "text",
            "content": block.get("text", ""),
        }

    elif block_type == "thinking":
        part = {
            "type": "reasoning",
            "content": block.get("thinking", ""),
        }

    elif block_type == "tool_use":
        part = {
            "type": "tool_call",
            "id": block.get("id", ""),
            "name": block.get("name", ""),
            "arguments": block.get("input", {}),
        }

    elif block_type == "tool_result":
        output = block.get("output", "")
        if isinstance(output, (list, dict)):
            result = _serialize_to_str(output)
        else:
            result = str(output)

        part = {
            "type": "tool_call_response",
            "id": block.get("id", ""),
            "response": result,
        }

    elif block_type == "image":
        source = block.get("source", {})
        source_type = source.get("type")

        if source_type == "url":
            url = source.get("url", "")
            part = {
                "type": "uri",
                "uri": url,
                "modality": "image",
            }
        elif source_type == "base64":
            data = source.get("data", "")
            media_type = source.get("media_type", "image/jpeg")
            part = {
                "type": "blob",
                "content": data,
                "media_type": media_type,
                "modality": "image",
            }
        else:
            part = None

    elif block_type == "audio":
        source = block.get("source", {})
        source_type = source.get("type")

        if source_type == "url":
            url = source.get("url", "")
            part = {
                "type": "uri",
                "uri": url,
                "modality": "audio",
            }
        elif source_type == "base64":
            data = source.get("data", "")
            media_type = source.get("media_type", "audio/wav")
            part = {
                "type": "blob",
                "content": data,
                "media_type": media_type,
                "modality": "audio",
            }
        else:
            part = None

    elif block_type == "video":
        source = block.get("source", {})
        source_type = source.get("type")

        if source_type == "url":
            url = source.get("url", "")
            part = {
                "type": "uri",
                "uri": url,
                "modality": "video",
            }
        elif source_type == "base64":
            data = source.get("data", "")
            media_type = source.get("media_type", "video/mp4")
            part = {
                "type": "blob",
                "content": data,
                "media_type": media_type,
                "modality": "video",
            }
        else:
            part = None

    else:
        part = None

    return part


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


def _get_provider_name(instance: Any) -> str:
    """Get provider name for the given instance.

    Maps AgentScope class names to OpenTelemetry GenAI provider names.

    Args:
        instance: Model, formatter, or embedding instance

    Returns:
        str: Standardized provider name or "unknown"
    """
    classname = instance.__class__.__name__
    return _PROVIDER_MAP.get(classname, "unknown")


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
        SpanAttributes.AGENTSCOPE_FUNCTION_METADATA: _serialize_to_str(
            {
                "model_name": getattr(instance, "model_name", "unknown_model"),
                "stream": getattr(instance, "stream", False),
            },
        ),


        # Old attributes
        SpanAttributes.OLD_SPAN_KIND: OldSpanKind.LLM,
        SpanAttributes.OLD_PROJECT_RUN_ID: _serialize_to_str(_config.run_id),
        SpanAttributes.OLD_INPUT: _serialize_to_str(
            {
                "args": args,
                "kwargs": kwargs,
            },
        ),
        SpanAttributes.OLD_META: _serialize_to_str(
            {
                "model_name": getattr(instance, "model_name", "unknown_model"),
                "stream": getattr(instance, "stream", False),
            },
        ),
    }

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

    # custom attributes
    attributes[SpanAttributes.AGENTSCOPE_FUNCTION_OUTPUT] = _serialize_to_str(
        chat_response,
    )

    attributes[SpanAttributes.OLD_OUTPUT] = _serialize_to_str(
        chat_response,
    )
    return attributes


def _convert_msg_to_parts(
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
        SpanAttributes.GEN_AI_SYSTEM_INSTRUCTIONS: instance.sys_prompt
        if hasattr(instance, "sys_prompt")
        else None,
    }

    msg = None
    if args and len(args) > 0:
        msg = args[0]
    elif "msg" in kwargs:
        msg = kwargs["msg"]
    if msg:
        input_messages = _convert_msg_to_parts(msg)
    else:
        input_messages = {
            "args": args,
            "kwargs": kwargs,
        }
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

    attributes[
        SpanAttributes.AGENTSCOPE_FUNCTION_METADATA
    ] = _serialize_to_str(
        {
            "name": getattr(instance, "name", "unknown_agent"),
            "id": getattr(instance, "id", "unknown"),
        },
    )

    # Old attributes
    attributes[SpanAttributes.OLD_SPAN_KIND] = OldSpanKind.AGENT

    attributes[SpanAttributes.OLD_PROJECT_RUN_ID] = _serialize_to_str(_config.run_id)
    attributes[SpanAttributes.OLD_INPUT] = _serialize_to_str(
            {
                "args": args,
                "kwargs": kwargs,
            },
    )
    attributes[SpanAttributes.OLD_META] = _serialize_to_str(
        {
            "id": getattr(instance, "id", "unknown"),
            "name": getattr(instance, "name", "unknown_agent"),
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
            _convert_msg_to_parts(agent_response),
        ),
        SpanAttributes.AGENTSCOPE_FUNCTION_OUTPUT: _serialize_to_str(
            agent_response,
        ),

        # Old attributes
        SpanAttributes.OLD_OUTPUT: _serialize_to_str(
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
        attributes[
            SpanAttributes.AGENTSCOPE_FUNCTION_METADATA
        ] = _serialize_to_str(
            {
                **tool_call,
            },
        )

        # Old attributes
        attributes[SpanAttributes.OLD_SPAN_KIND] = OldSpanKind.TOOL
        attributes[SpanAttributes.OLD_PROJECT_RUN_ID] = _serialize_to_str(_config.run_id)
        attributes[
            SpanAttributes.OLD_INPUT
        ] = _serialize_to_str(
            {
                "tool_call": tool_call,
            },
        )
        attributes[
            SpanAttributes.OLD_META
        ] = _serialize_to_str(
            {
                **tool_call,
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

    # Old attributes
    attributes[SpanAttributes.OLD_OUTPUT] = _serialize_to_str(
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
        SpanAttributes.GEN_AI_PROVIDER_NAME: _get_provider_name(instance),
        SpanAttributes.AGENTSCOPE_FORMAT_INPUT: _serialize_to_str(
            {
                "args": args,
                "kwargs": kwargs,
            },
        ),
        SpanAttributes.AGENTSCOPE_FUNCTION_INPUT: _serialize_to_str(
            {
                "args": args,
                "kwargs": kwargs,
            },
        ),

        # Old attributes
        SpanAttributes.OLD_SPAN_KIND: OldSpanKind.FORMATTER,
        SpanAttributes.OLD_PROJECT_RUN_ID: _serialize_to_str(_config.run_id),
        SpanAttributes.OLD_INPUT: _serialize_to_str(
            {
                "args": args,
                "kwargs": kwargs,
            },
        ),

        # Need to delete
        SpanAttributes.OLD_META: _serialize_to_str({}),
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
        f"{attributes[SpanAttributes.GEN_AI_PROVIDER_NAME]}"
    )


def get_formatter_response_attributes(
    response: Any,
) -> Dict[str, str]:
    """Get formatter response attributes for OpenTelemetry tracing.

    Args:
        response: Response object from formatter

    Returns:
        Dict[str, str]: OpenTelemetry GenAI response attributes
    """
    attributes = {
        SpanAttributes.AGENTSCOPE_FORMAT_OUTPUT: _serialize_to_str(response),
        SpanAttributes.AGENTSCOPE_FUNCTION_OUTPUT: _serialize_to_str(response),
        # Old attributes
        SpanAttributes.OLD_OUTPUT: _serialize_to_str(response),
    }
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
        # Old attributes
        SpanAttributes.OLD_SPAN_KIND: OldSpanKind.COMMON,
        SpanAttributes.OLD_PROJECT_RUN_ID: _serialize_to_str(_config.run_id),
        SpanAttributes.OLD_INPUT: _serialize_to_str(
            {
                "args": args,
                "kwargs": kwargs,
            },
        ),
        SpanAttributes.OLD_META: _serialize_to_str({})
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
        # Old attributes
        SpanAttributes.OLD_OUTPUT: _serialize_to_str(response),
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

        # Old attributes
        SpanAttributes.OLD_SPAN_KIND: OldSpanKind.EMBEDDING,
        SpanAttributes.OLD_PROJECT_RUN_ID: _serialize_to_str(_config.run_id),
        SpanAttributes.OLD_INPUT: _serialize_to_str(
            {
                "args": args,
                "kwargs": kwargs,
            },
        ),
        SpanAttributes.OLD_META: _serialize_to_str(
            {
                "model_name": getattr(instance, "model_name", "unknown_model"),
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
        # Old attributes
        SpanAttributes.OLD_OUTPUT: _serialize_to_str(response),
    }
    return attributes
