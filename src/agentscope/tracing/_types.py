# -*- coding: utf-8 -*-
"""The tracing types class in agentscope."""
from enum import Enum

class SpanAttributes:
    """The span attributes."""

    SPAN_KIND = "span.kind"
    OUTPUT = "output"
    INPUT = "input"
    META = "metadata"
    PROJECT_RUN_ID = "project.run_id"

    GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
    GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"

    AGENTSCOPE_FORMAT_INPUT = "agentscope.format.input"
    AGENTSCOPE_FORMAT_OUTPUT = "agentscope.format.output"

    AGENTSCOPE_FUNCTION_NAME = "agentscope.function.name"
    AGENTSCOPE_FUNCTION_INPUT = "agentscope.function.input"
    AGENTSCOPE_FUNCTION_OUTPUT = "agentscope.function.output"

    AGENTSCOPE_INPUT = "agentscope.input"
    AGENTSCOPE_OUTPUT = "agentscope.output"


class OperationNameValues(str, Enum):
    """The provider name values."""
    FORMATTER = "format"
    INVOKE_GENERIC_FUNCTION = "invoke_generic_function"


class ProviderNameValues(str, Enum):
    """The provider name values."""
    DASHSCOPE = "dashscope"
    OLLAMA = "ollama"
