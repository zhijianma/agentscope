# -*- coding: utf-8 -*-
"""Message converter utilities for OpenTelemetry tracing.

This module provides utilities to convert messages from different formats
(AgentScope Msg objects, formatter outputs) into OpenTelemetry GenAI format.
"""
from typing import Any, Optional, TYPE_CHECKING

from ._utils import _serialize_to_str

if TYPE_CHECKING:
    from ..message import (
        ContentBlock,
    )
else:
    ContentBlock = "ContentBlock"


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
