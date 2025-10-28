import inspect
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


from ._attributes import _serialize_to_str
from .. import _config
from ..embedding._embedding_base import EmbeddingModelBase
from ..model._model_base import ChatModelBase
from .._logging import logger
from ._types import SpanAttributes, OperationNameValues, ProviderNameValues

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


_PROVIDER_MAP = {

    # LLM Models
    "OpenAIChatModel": GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
    "GeminiChatModel": GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value,
    "AnthropicChatModel": GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value,
    "DashScopeChatModel": ProviderNameValues.DASHSCOPE.value,
    "OllamaChatModel": ProviderNameValues.OLLAMA.value,

    # Formatter
    "DashScopeChatFormatter": ProviderNameValues.DASHSCOPE.value,
    "DashScopeMultiAgentFormatter": ProviderNameValues.DASHSCOPE.value,
    "OpenAIChatFormatter": GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
    "OpenAIMultiAgentFormatter": GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
    "AnthropicChatFormatter": GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value,
    "AnthropicMultiAgentFormatter": GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value,
    "GeminiChatFormatter": GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value,
    "GeminiMultiAgentFormatter": GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value,
    "OllamaChatFormatter": ProviderNameValues.OLLAMA.value,
    "OllamaMultiAgentFormatter": ProviderNameValues.OLLAMA.value,
    "DeepSeekChatFormatter": GenAIAttributes.GenAiProviderNameValues.DEEPSEEK.value,
    "DeepSeekMultiAgentFormatter": GenAIAttributes.GenAiProviderNameValues.DEEPSEEK.value,

    # Embedding Models
    "DashScopeTextEmbedding": ProviderNameValues.DASHSCOPE.value,
    "DashScopeMultiModalEmbedding": ProviderNameValues.DASHSCOPE.value,
    "OpenAITextEmbedding": GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
    "GeminiTextEmbedding": GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value,
    "OllamaTextEmbedding": ProviderNameValues.OLLAMA.value,
}

def _convert_block_to_part(block: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Convert a block to a part according to OpenTelemetry GenAI schema.

    This function converts different types of content blocks (text, thinking, tool_use,
    tool_result, image, audio, video) into standardized parts that conform to the
    OpenTelemetry GenAI semantic conventions.

    Args:
        block: A dictionary containing the block data with at least a "type" key.
               Supported types: "text", "thinking", "tool_use", "tool_result",
               "image", "audio", "video".

    Returns:
        Optional[dict[str, Any]]: A standardized part dictionary with type-specific
                                  fields, or None if the block type is not supported.

    Examples:
        >>> block = {"type": "text", "text": "Hello world"}
        >>> _convert_block_to_part(block)
        {"type": "text", "content": "Hello world"}

        >>> block = {"type": "tool_use", "id": "call_1", "name": "search", "input": {"query": "test"}}
        >>> _convert_block_to_part(block)
        {"type": "tool_call", "id": "call_1", "name": "search", "arguments": {"query": "test"}}
    """
    block_type = block.get("type")

    if block_type == "text":
        part = {
            "type": "text",
            "content": block.get("text", "")
        }

    elif block_type == "thinking":
        part = {
            "type": "reasoning",
            "content": block.get("thinking", "")
        }

    elif block_type == "tool_use":
        part = {
            "type": "tool_call",
            "id": block.get("id", ""),
            "name": block.get("name", ""),
            "arguments": block.get("input", {})
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
            "response": result
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

    This function returns attributes that are common to all tracing spans,
    such as the conversation ID derived from the current run context.

    Returns:
        Dict[str, str]: A dictionary containing common span attributes.
                       Currently includes the GenAI conversation ID.

    Note:
        The conversation ID is derived from the global run_id configuration.
    """
    return {
        GenAIAttributes.GEN_AI_CONVERSATION_ID: _serialize_to_str(_config.run_id),
    }

def _get_provider_name(instance: Any) -> str:
    """Get the provider name for the given instance.

    Maps AgentScope class names to their corresponding OpenTelemetry GenAI
    provider names. This is used to standardize provider identification
    across different model and formatter implementations.

    Args:
        instance: The model, formatter, or embedding instance to get the
                  provider name for.

    Returns:
        str: The standardized provider name (e.g., "openai", "anthropic",
             "dashscope"), or "unknown" if the class is not mapped.

    Note:
        The mapping is based on the class name of the instance, not the
        actual provider configuration.
    """
    classname = instance.__class__.__name__
    return _PROVIDER_MAP.get(classname, "unknown")

def get_llm_request_attributes(
    instance: ChatModelBase,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> Dict[str, str]:
    """Get the LLM request attributes for OpenTelemetry tracing.

    Extracts and formats request parameters from LLM model calls into
    standardized OpenTelemetry GenAI attributes. This includes model
    configuration, generation parameters, and input data.

    Args:
        instance: The ChatModelBase instance making the request.
        args: Positional arguments passed to the model call.
        kwargs: Keyword arguments passed to the model call, including
                generation parameters like temperature, top_p, max_tokens, etc.

    Returns:
        Dict[str, str]: A dictionary of OpenTelemetry GenAI attributes
                       with string values.

    Note:
        Only non-None values are included in the returned dictionary.
        Input data is serialized using the _serialize_to_str function.
    """

    attributes = {
        # required attributes
        GenAIAttributes.GEN_AI_OPERATION_NAME: GenAIAttributes.GenAiOperationNameValues.CHAT.value,
        GenAIAttributes.GEN_AI_PROVIDER_NAME: _get_provider_name(instance),

        # conditionally required attributes
        GenAIAttributes.GEN_AI_REQUEST_MODEL: getattr(instance, "model_name", "unknown_model"),

        # recommended attributes
        GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE: kwargs.get("temperature"),
        GenAIAttributes.GEN_AI_REQUEST_TOP_P: kwargs.get("p")
        or kwargs.get("top_p"),
        GenAIAttributes.GEN_AI_REQUEST_TOP_K: kwargs.get("top_k"),
        GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS: kwargs.get("max_tokens"),
        GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY: kwargs.get(
            "presence_penalty"
        ),
        GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY: kwargs.get(
            "frequency_penalty"
        ),
        GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES: kwargs.get("stop_sequences"),
        GenAIAttributes.GEN_AI_REQUEST_SEED: kwargs.get("seed"),
        SpanAttributes.AGENTSCOPE_FUNCTION_INPUT: _serialize_to_str(
            {
                "args": args,
                "kwargs": kwargs,
            },
        ),
    }

    return {k: v for k, v in attributes.items() if v is not None}

def get_llm_span_name(attributes: Dict[str, str]) -> str:
    """Generate a human-readable span name for LLM operations.

    Creates a descriptive span name by combining the operation name
    and model name from the attributes dictionary.

    Args:
        attributes: Dictionary containing LLM request attributes,
                   must include GEN_AI_OPERATION_NAME and GEN_AI_REQUEST_MODEL.

    Returns:
        str: A formatted span name in the format "{operation} {model}".

    Raises:
        KeyError: If required attributes are missing from the input dictionary.
    """
    return f"{attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]} {attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]}"


def _get_llm_output_messages(
    chat_response: Any,
) -> list[dict[str, Any]]:
    """Extract and format LLM output messages for OpenTelemetry tracing.

    Converts AgentScope ChatResponse objects into standardized message
    format suitable for OpenTelemetry GenAI attributes. Handles different
    content block types and provides error handling for malformed responses.

    Args:
        chat_response: The chat response object, typically a ChatResponse
                      instance containing content blocks.

    Returns:
        list[dict[str, Any]]: A list containing a single formatted message

    Note:
        If the response is not a ChatResponse instance or processing fails,
        returns a single error message with appropriate error handling.
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
            "finish_reason": finish_reason
        }

        return [output_message]

    except Exception as e:
        return [{
            "role": "assistant",
            "parts": [{
                "type": "text",
                "content": "<error processing response>"
            }],
            "finish_reason": "error"
        }]


def get_llm_response_attributes(
    chat_response: Any,
) -> Dict[str, str]:
    """Get the LLM response attributes for OpenTelemetry tracing.

    Extracts response metadata from LLM model responses and formats them
    into OpenTelemetry GenAI attributes. This includes response ID,
    usage statistics, and output messages.

    Args:
        chat_response: The chat response object containing response data,
                      usage information, and content.

    Returns:
        Dict[str, str]: A dictionary of OpenTelemetry GenAI response attributes

    Note:
        Usage statistics are only included if the response has a usage attribute.
        All complex data structures are serialized to strings.
    """
    attributes = {
        GenAIAttributes.GEN_AI_RESPONSE_ID: getattr(chat_response, "id", "unknown_id"),
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS: '["stop"]'
    }
    if hasattr(chat_response, "usage") and chat_response.usage:
        attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = chat_response.usage.input_tokens
        attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = chat_response.usage.output_tokens

    output_messages = _get_llm_output_messages(chat_response)
    if output_messages:
        attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES] = _serialize_to_str(output_messages)
    attributes[SpanAttributes.AGENTSCOPE_FUNCTION_OUTPUT] = _serialize_to_str(chat_response)
    return attributes


def _convert_msg_to_parts(
    msg: Msg
) -> dict[str, Any]:
    """Convert an AgentScope message to standardized parts format.

    Transforms AgentScope Msg objects into a format suitable for
    OpenTelemetry GenAI attributes by converting all content blocks
    to standardized parts.

    Args:
        msg: The AgentScope message object containing content blocks
             and metadata.

    Returns:
        dict[str, Any]: A formatted message dictionary containing:

    Note:
        If conversion fails, returns a fallback message with basic
        text content. Debug logging is used for error tracking.
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
            "parts": parts
        }

        if msg.name:
            formatted_msg["name"] = msg.name

        return formatted_msg

    except Exception as e:
        logger.debug(f"Error formatting message: {e}")
        return {
            "role": msg.role,
            "parts": [{
                "type": "text",
                "content": str(msg.content) if msg.content else ""
            }]
        }

def get_agent_request_attributes(
        instance: "AgentBase",
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any]
    ) -> Dict[str, str]:
    """Get the agent request attributes for OpenTelemetry tracing.

    Extracts agent-specific metadata and input data from agent invocation
    calls and formats them into OpenTelemetry GenAI attributes.

    Args:
        instance: The AgentBase instance being invoked.
        args: Positional arguments passed to the agent call.
        kwargs: Keyword arguments passed to the agent call.

    Returns:
        Dict[str, str]: A dictionary of OpenTelemetry GenAI agent attributes

    Note:
        The first argument or "msg" keyword argument is treated as
        the primary input message and converted to parts format.
    """
    attributes = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: GenAIAttributes.GenAiOperationNameValues.INVOKE_AGENT.value,
        GenAIAttributes.GEN_AI_AGENT_ID: getattr(instance, "id", "unknown"),
        GenAIAttributes.GEN_AI_AGENT_NAME: getattr(instance, "name", "unknown_agent"),
        GenAIAttributes.GEN_AI_AGENT_DESCRIPTION: inspect.getdoc(instance.__class__) or "No description available",
        GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS: instance.sys_prompt if hasattr(instance, "sys_prompt") else None,
    }

    if hasattr(instance, "model") and instance.model:
        attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] = getattr(instance.model, "model_name", "unknown_model")

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
    attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES] = _serialize_to_str(input_messages)

    attributes[SpanAttributes.AGENTSCOPE_FUNCTION_INPUT] = _serialize_to_str({
        "args": args,
        "kwargs": kwargs,
    })

    return attributes

def get_agent_span_name(attributes: Dict[str, str]) -> str:
    """Generate a human-readable span name for agent operations.

    Creates a descriptive span name by combining the operation name
    and agent name from the attributes dictionary.

    Args:
        attributes: Dictionary containing agent request attributes,
                   must include GEN_AI_OPERATION_NAME and GEN_AI_AGENT_NAME.

    Returns:
        str: A formatted span name in the format "{operation} {agent_name}".

    Raises:
        KeyError: If required attributes are missing from the input dictionary.
    """
    return f"{attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]} {attributes[GenAIAttributes.GEN_AI_AGENT_NAME]}"

def get_agent_response_attributes(
    agent_response: Any,
) -> Dict[str, str]:
    """Get the agent response attributes for OpenTelemetry tracing.

    Formats agent response data into OpenTelemetry GenAI attributes.
    This typically includes the output message and raw response data.

    Args:
        agent_response: The response object returned by the agent,
                       typically a Msg object.

    Returns:
        Dict[str, str]: A dictionary of OpenTelemetry GenAI response attributes

    Note:
        The agent response is converted to parts format and serialized
        for inclusion in tracing attributes.
    """
    attributes = {
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES: _serialize_to_str(_convert_msg_to_parts(agent_response)),
        SpanAttributes.AGENTSCOPE_FUNCTION_OUTPUT: _serialize_to_str(agent_response),
    }
    return attributes


def get_tool_request_attributes(
    instance: "Toolkit",
    tool_call: ToolUseBlock,
) -> Dict[str, str]:
    """Get the tool request attributes for OpenTelemetry tracing.

    Extracts tool execution metadata from tool calls and formats them
    into OpenTelemetry GenAI attributes. This includes tool identification,
    arguments, and description information.

    Args:
        instance: The Toolkit instance containing the tool definitions.
        tool_call: The tool use block containing tool call information
                  including tool name, ID, and input arguments.

    Returns:
        Dict[str, str]: A dictionary of OpenTelemetry GenAI tool attributes

    Note:
        Tool description is extracted from the tool's JSON schema if available.
        All complex data structures are serialized to strings.
    """
    attributes = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: GenAIAttributes.GenAiOperationNameValues.EXECUTE_TOOL.value,
    }

    if tool_call:
        tool_name = tool_call.get("name")
        attributes[GenAIAttributes.GEN_AI_TOOL_CALL_ID] = tool_call.get("id")
        attributes[GenAIAttributes.GEN_AI_TOOL_NAME] = tool_name
        attributes[SpanAttributes.GEN_AI_TOOL_CALL_ARGUMENTS] = _serialize_to_str(tool_call.get("input"))

        if tool_name:
            if tool := getattr(instance, 'tools', {}).get(tool_name):
                if tool_func := getattr(tool, "json_schema", {}).get("function", {}):
                    attributes[GenAIAttributes.GEN_AI_TOOL_DESCRIPTION] = tool_func.get("description", "unknown_description")


        attributes[SpanAttributes.AGENTSCOPE_FUNCTION_INPUT] = _serialize_to_str(
                    {
                        "tool_call": tool_call,
                    },
        )
    return attributes

def get_tool_span_name(attributes: Dict[str, str]) -> str:
    """Generate a human-readable span name for tool operations.

    Creates a descriptive span name by combining the operation name
    and tool name from the attributes dictionary.

    Args:
        attributes: Dictionary containing tool request attributes,
                   must include GEN_AI_OPERATION_NAME and GEN_AI_TOOL_NAME.

    Returns:
        str: A formatted span name in the format "{operation} {tool_name}".

    Raises:
        KeyError: If required attributes are missing from the input dictionary.
    """
    return f"{attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]} {attributes[GenAIAttributes.GEN_AI_TOOL_NAME]}"


def get_tool_response_attributes(
    tool_response: Any,
) -> Dict[str, str]:
    """Get the tool response attributes for OpenTelemetry tracing.

    Formats tool execution results into OpenTelemetry GenAI attributes.
    This includes the tool call result and raw response data.

    Args:
        tool_response: The response object returned by the tool execution,
                      typically containing the tool's output or result.

    Returns:
        Dict[str, str]: A dictionary of OpenTelemetry GenAI response attributes

    Note:
        All response data is serialized to strings for inclusion in
        tracing attributes.
    """
    attributes = {
        SpanAttributes.GEN_AI_TOOL_CALL_RESULT: _serialize_to_str(tool_response),
    }

    attributes[SpanAttributes.AGENTSCOPE_FUNCTION_OUTPUT] = _serialize_to_str(tool_response)

    return attributes


def get_formatter_request_attributes(
    instance: "FormatterBase",
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> Dict[str, str]:
    """Get the formatter request attributes for OpenTelemetry tracing.

    Extracts formatter-specific metadata from formatter calls and formats
    them into OpenTelemetry GenAI attributes. This includes provider
    information and input data.

    Args:
        instance: The FormatterBase instance being used.
        args: Positional arguments passed to the formatter call.
        kwargs: Keyword arguments passed to the formatter call.

    Returns:
        Dict[str, str]: A dictionary of OpenTelemetry GenAI formatter attributes

    Note:
        All input data is serialized to strings for inclusion in
        tracing attributes.
    """
    attributes = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: OperationNameValues.FORMATTER.value,
        GenAIAttributes.GEN_AI_PROVIDER_NAME: _get_provider_name(instance),
        SpanAttributes.AGENTSCOPE_FORMAT_INPUT: _serialize_to_str(
            {
                "args": args,
                "kwargs": kwargs,
            },
        ),
    }
    return attributes


def get_formatter_span_name(attributes: Dict[str, str]) -> str:
    """Generate a human-readable span name for formatter operations.

    Creates a descriptive span name by combining the operation name
    and provider name from the attributes dictionary.

    Args:
        attributes: Dictionary containing formatter request attributes,
                   must include GEN_AI_OPERATION_NAME and GEN_AI_PROVIDER_NAME.

    Returns:
        str: A formatted span name in the format "{operation} {provider}".
             Example: "formatter openai" or "formatter dashscope".
    """
    return f"{attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]} {attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME]}"

def get_formatter_response_attributes(
    response: Any,
) -> Dict[str, str]:
    """Get the formatter response attributes for OpenTelemetry tracing.

    Formats formatter output data into OpenTelemetry GenAI attributes.
    This includes both the formatted output and raw response data.

    Args:
        response: The response object returned by the formatter,
                 typically containing formatted data.

    Returns:
        Dict[str, str]: A dictionary of OpenTelemetry GenAI response attributes

    Note:
        Both the raw response and formatted output are included
        for comprehensive tracing coverage.
    """
    attributes = {
        SpanAttributes.AGENTSCOPE_FORMAT_OUTPUT: _serialize_to_str(response),
    }
    return attributes


def get_generic_function_request_attributes(
    function_name: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> Dict[str, str]:
    """Get the generic function request attributes for OpenTelemetry tracing.

    Extracts metadata from generic function calls and formats them
    into OpenTelemetry GenAI attributes. This is used for tracing
    arbitrary function invocations within the AgentScope framework.

    Args:
        function_name: The name of the function being called.
        args: Positional arguments passed to the function call.
        kwargs: Keyword arguments passed to the function call.

    Returns:
        Dict[str, str]: A dictionary of OpenTelemetry GenAI function attributes
    Note:
        This function is used for tracing generic function calls that
        don't fall into specific categories like LLM, agent, or tool calls.
    """
    attributes = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: OperationNameValues.INVOKE_GENERIC_FUNCTION.value,
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
    """Generate a human-readable span name for generic function operations.

    Creates a descriptive span name by combining the operation name
    and function name from the attributes dictionary.

    Args:
        attributes: Dictionary containing generic function request attributes,
                   must include GEN_AI_OPERATION_NAME and AGENTSCOPE_FUNCTION_NAME.

    Returns:
        str: A formatted span name in the format "{operation} {function_name}".
    """
    return f"{attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]} {attributes[SpanAttributes.AGENTSCOPE_FUNCTION_NAME]}"

def get_generic_function_response_attributes(
    response: Any,
) -> Dict[str, str]:
    """Get the generic function response attributes for OpenTelemetry tracing.

    Formats generic function output data into OpenTelemetry GenAI attributes.
    This includes both the function output and raw response data.

    Args:
        response: The response object returned by the generic function,
                 containing the function's output or result.

    Returns:
        Dict[str, str]: A dictionary of OpenTelemetry GenAI response attributes

    Note:
        All response data is serialized to strings for inclusion in
        tracing attributes.
    """
    attributes = {
        SpanAttributes.AGENTSCOPE_FUNCTION_OUTPUT: _serialize_to_str(response),
    }
    return attributes


def get_embedding_request_attributes(
    instance: "EmbeddingModelBase",
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> Dict[str, str]:
    """Get the embedding request attributes for OpenTelemetry tracing.

    Extracts embedding model metadata and input data from embedding
    requests and formats them into OpenTelemetry GenAI attributes.

    Args:
        instance: The EmbeddingModelBase instance making the request.
        args: Positional arguments passed to the embedding call.
        kwargs: Keyword arguments passed to the embedding call,
                including optional parameters like dimensions.

    Returns:
        Dict[str, str]: A dictionary of OpenTelemetry GenAI embedding attributes

    Note:
        The dimensions parameter is extracted from kwargs if present.
        All input data is serialized to strings for inclusion in
        tracing attributes.
    """
    attributes = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value,
        GenAIAttributes.GEN_AI_REQUEST_MODEL: getattr(instance, "model_name", "unknown_model"),
        GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT: kwargs.get("dimensions"),

        SpanAttributes.AGENTSCOPE_FUNCTION_INPUT: _serialize_to_str(
                {
                    "args": args,
                    "kwargs": kwargs,
                },
            ),
    }
    return attributes

def get_embedding_span_name(attributes: Dict[str, str]) -> str:
    """Generate a human-readable span name for embedding operations.

    Creates a descriptive span name by combining the operation name
    and model name from the attributes dictionary.

    Args:
        attributes: Dictionary containing embedding request attributes,
                   must include GEN_AI_OPERATION_NAME and GEN_AI_REQUEST_MODEL.

    Returns:
        str: A formatted span name in the format "{operation} {model}".
    """
    return f"{attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]} {attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]}"

def get_embedding_response_attributes(
    response: Any,
) -> Dict[str, str]:
    """Get the embedding response attributes for OpenTelemetry tracing.

    Formats embedding model output data into OpenTelemetry GenAI attributes.
    This includes the embedding vectors and any associated metadata.

    Args:
        response: The response object returned by the embedding model,
                 typically containing embedding vectors and metadata.

    Returns:
        Dict[str, str]: A dictionary of OpenTelemetry GenAI response attributes

    Note:
        The response data is serialized to strings for inclusion in
        tracing attributes. This typically includes embedding vectors
        and any associated metadata from the embedding model.
    """
    attributes = {
        SpanAttributes.AGENTSCOPE_FUNCTION_OUTPUT: _serialize_to_str(response),
    }
    return attributes
