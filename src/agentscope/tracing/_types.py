# -*- coding: utf-8 -*-
"""The tracing types class in agentscope."""
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


class SpanAttributes:
    """The span attributes."""

    GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
    """The gen ai tool call arguments."""

    GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
    """The gen ai tool call result."""

    AGENTSCOPE_FORMAT_INPUT = "agentscope.format.input"
    """The agentscope format input."""

    AGENTSCOPE_FORMAT_OUTPUT = "agentscope.format.output"
    """The agentscope format output."""

    AGENTSCOPE_FUNCTION_NAME = "agentscope.function.name"
    """The agentscope function name."""

    AGENTSCOPE_FUNCTION_INPUT = "agentscope.function.input"
    """The agentscope function input."""

    AGENTSCOPE_FUNCTION_OUTPUT = "agentscope.function.output"
    """The agentscope function output."""

    AGENTSCOPE_FUNCTION_METADATA = "agentscope.function.metadata"
    """The agentscope function metadata."""


class OperationNameValues:
    """The operation name values."""

    FORMATTER = "format"
    """The formatter operation name."""

    INVOKE_GENERIC_FUNCTION = "invoke_generic_function"
    """The invoke generic function operation name."""

    CHAT_MODEL = "chat_model"
    """The chat model operation name."""

    CHAT = GenAIAttributes.GenAiOperationNameValues.CHAT.value
    """The chat operation name."""

    INVOKE_AGENT = GenAIAttributes.GenAiOperationNameValues.INVOKE_AGENT.value
    """The invoke agent operation name."""

    EXECUTE_TOOL = GenAIAttributes.GenAiOperationNameValues.EXECUTE_TOOL.value
    """The execute tool operation name."""

    EMBEDDINGS = GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value
    """The embeddings operation name."""


class ProviderNameValues:
    """The provider name values."""

    DASHSCOPE = "dashscope"
    """The dashscope provider name."""

    OLLAMA = "ollama"
    """The ollama provider name."""

    DEEPSEEK = GenAIAttributes.GenAiProviderNameValues.DEEPSEEK.value
    """The deepseek provider name."""

    OPENAI = GenAIAttributes.GenAiProviderNameValues.OPENAI.value
    """The openai provider name."""

    ANTHROPIC = GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value
    """The anthropic provider name."""

    GCP_GEMINI = GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value
    """The gcp gemini provider name."""
