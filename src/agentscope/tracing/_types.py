# -*- coding: utf-8 -*-
"""The tracing types class in agentscope."""
from enum import Enum

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


class OperationNameValues(str, Enum):
    """The provider name values."""

    FORMATTER = "format"
    """The formatter operation name."""

    INVOKE_GENERIC_FUNCTION = "invoke_generic_function"
    """The invoke generic function operation name."""


class ProviderNameValues(str, Enum):
    """The provider name values."""

    DASHSCOPE = "dashscope"
    """The dashscope provider name."""

    OLLAMA = "ollama"
    """The ollama provider name."""
