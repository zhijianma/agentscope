# -*- coding: utf-8 -*-
"""Unittests for the message converter functionality in AgentScope tracing."""
import sys
from pathlib import Path
from typing import Any, Dict
from unittest import TestCase

# Add src to path for development testing
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from agentscope.message import (
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ImageBlock,
    AudioBlock,
    VideoBlock,
)
from agentscope.tracing._converter import _convert_block_to_part


class ConverterTest(TestCase):
    """Test cases for _convert_block_to_part converter functionality"""

    def test_convert_text_block(self) -> None:
        """Test converting text block to OpenTelemetry GenAI format"""
        block: TextBlock = {
            "type": "text",
            "text": "Hello, world!",
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "text")
        self.assertEqual(result["content"], "Hello, world!")

    def test_convert_text_block_missing_text(self) -> None:
        """Test converting text block with missing text field"""
        block: TextBlock = {
            "type": "text",
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "text")
        self.assertEqual(result["content"], "")

    def test_convert_text_block_empty_text(self) -> None:
        """Test converting text block with empty text"""
        block: TextBlock = {
            "type": "text",
            "text": "",
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "text")
        self.assertEqual(result["content"], "")

    def test_convert_thinking_block(self) -> None:
        """Test converting thinking block to reasoning format"""
        block: Dict[str, Any] = {
            "type": "thinking",
            "thinking": "Let me think about this...",
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "reasoning")
        self.assertEqual(result["content"], "Let me think about this...")

    def test_convert_thinking_block_missing_thinking(self) -> None:
        """Test converting thinking block with missing thinking field"""
        block: Dict[str, Any] = {
            "type": "thinking",
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "reasoning")
        self.assertEqual(result["content"], "")

    def test_convert_tool_use_block(self) -> None:
        """Test converting tool_use block to tool_call format"""
        block: ToolUseBlock = {
            "type": "tool_use",
            "id": "call_123",
            "name": "search",
            "input": {"query": "test query", "limit": 10},
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "tool_call")
        self.assertEqual(result["id"], "call_123")
        self.assertEqual(result["name"], "search")
        self.assertEqual(result["arguments"], {"query": "test query", "limit": 10})

    def test_convert_tool_use_block_missing_fields(self) -> None:
        """Test converting tool_use block with missing optional fields"""
        block: ToolUseBlock = {
            "type": "tool_use",
            "id": "",
            "name": "",
            "input": {},
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "tool_call")
        self.assertEqual(result["id"], "")
        self.assertEqual(result["name"], "")
        self.assertEqual(result["arguments"], {})

    def test_convert_tool_result_block_string_output(self) -> None:
        """Test converting tool_result block with string output"""
        block: ToolResultBlock = {
            "type": "tool_result",
            "id": "call_123",
            "name": "search",
            "output": "Search results here",
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "tool_call_response")
        self.assertEqual(result["id"], "call_123")
        self.assertEqual(result["response"], "Search results here")

    def test_convert_tool_result_block_dict_output(self) -> None:
        """Test converting tool_result block with dict output"""
        block: ToolResultBlock = {
            "type": "tool_result",
            "id": "call_123",
            "name": "search",
            "output": {"result": "success", "data": [1, 2, 3]},
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "tool_call_response")
        self.assertEqual(result["id"], "call_123")
        # Dict should be serialized to JSON string
        self.assertIsInstance(result["response"], str)
        self.assertIn("result", result["response"])
        self.assertIn("success", result["response"])

    def test_convert_tool_result_block_list_output(self) -> None:
        """Test converting tool_result block with list output"""
        block: ToolResultBlock = {
            "type": "tool_result",
            "id": "call_123",
            "name": "search",
            "output": [
                {"type": "text", "text": "Result 1"},
                {"type": "text", "text": "Result 2"},
            ],
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "tool_call_response")
        self.assertEqual(result["id"], "call_123")
        # List should be serialized to JSON string
        self.assertIsInstance(result["response"], str)
        self.assertIn("Result 1", result["response"])

    def test_convert_tool_result_block_numeric_output(self) -> None:
        """Test converting tool_result block with numeric output"""
        block: ToolResultBlock = {
            "type": "tool_result",
            "id": "call_123",
            "name": "calculate",
            "output": 42,
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "tool_call_response")
        self.assertEqual(result["id"], "call_123")
        self.assertEqual(result["response"], "42")

    def test_convert_tool_result_block_missing_id(self) -> None:
        """Test converting tool_result block with missing id"""
        block: ToolResultBlock = {
            "type": "tool_result",
            "id": "",
            "name": "search",
            "output": "result",
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "tool_call_response")
        self.assertEqual(result["id"], "")

    def test_convert_image_block_url_source(self) -> None:
        """Test converting image block with URL source"""
        source: Dict[str, Any] = {
            "type": "url",
            "url": "https://example.com/image.jpg",
        }
        block: ImageBlock = {
            "type": "image",
            "source": source,
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "uri")
        self.assertEqual(result["uri"], "https://example.com/image.jpg")
        self.assertEqual(result["modality"], "image")

    def test_convert_image_block_base64_source(self) -> None:
        """Test converting image block with base64 source"""
        source: Dict[str, Any] = {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        }
        block: ImageBlock = {
            "type": "image",
            "source": source,
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "blob")
        self.assertEqual(result["content"], source["data"])
        self.assertEqual(result["media_type"], "image/png")
        self.assertEqual(result["modality"], "image")

    def test_convert_image_block_base64_default_media_type(self) -> None:
        """Test converting image block with base64 source using default media_type"""
        source: Dict[str, Any] = {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "base64data",
        }
        block: ImageBlock = {
            "type": "image",
            "source": source,
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "blob")
        self.assertEqual(result["media_type"], "image/jpeg")

    def test_convert_image_block_invalid_source_type(self) -> None:
        """Test converting image block with invalid source type"""
        block: ImageBlock = {
            "type": "image",
            "source": {
                "type": "invalid",
            },
        }
        result = _convert_block_to_part(block)

        self.assertIsNone(result)

    def test_convert_image_block_missing_source(self) -> None:
        """Test converting image block with missing source"""
        block = {
            "type": "image",
        }
        result = _convert_block_to_part(block)

        # Should return None when source type is not url or base64
        self.assertIsNone(result)

    def test_convert_audio_block_url_source(self) -> None:
        """Test converting audio block with URL source"""
        source: Dict[str, Any] = {
            "type": "url",
            "url": "https://example.com/audio.wav",
        }
        block: AudioBlock = {
            "type": "audio",
            "source": source,
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "uri")
        self.assertEqual(result["uri"], "https://example.com/audio.wav")
        self.assertEqual(result["modality"], "audio")

    def test_convert_audio_block_base64_source(self) -> None:
        """Test converting audio block with base64 source"""
        source: Dict[str, Any] = {
            "type": "base64",
            "media_type": "audio/mpeg",
            "data": "base64audiodata",
        }
        block: AudioBlock = {
            "type": "audio",
            "source": source,
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "blob")
        self.assertEqual(result["content"], "base64audiodata")
        self.assertEqual(result["media_type"], "audio/mpeg")
        self.assertEqual(result["modality"], "audio")

    def test_convert_audio_block_base64_default_media_type(self) -> None:
        """Test converting audio block with base64 source using default media_type"""
        source: Dict[str, Any] = {
            "type": "base64",
            "media_type": "audio/wav",
            "data": "base64data",
        }
        block: AudioBlock = {
            "type": "audio",
            "source": source,
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "blob")
        self.assertEqual(result["media_type"], "audio/wav")

    def test_convert_audio_block_invalid_source_type(self) -> None:
        """Test converting audio block with invalid source type"""
        block: AudioBlock = {
            "type": "audio",
            "source": {
                "type": "invalid",
            },
        }
        result = _convert_block_to_part(block)

        self.assertIsNone(result)

    def test_convert_video_block_url_source(self) -> None:
        """Test converting video block with URL source"""
        source: Dict[str, Any] = {
            "type": "url",
            "url": "https://example.com/video.mp4",
        }
        block: VideoBlock = {
            "type": "video",
            "source": source,
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "uri")
        self.assertEqual(result["uri"], "https://example.com/video.mp4")
        self.assertEqual(result["modality"], "video")

    def test_convert_video_block_base64_source(self) -> None:
        """Test converting video block with base64 source"""
        source: Dict[str, Any] = {
            "type": "base64",
            "media_type": "video/webm",
            "data": "base64videodata",
        }
        block: VideoBlock = {
            "type": "video",
            "source": source,
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "blob")
        self.assertEqual(result["content"], "base64videodata")
        self.assertEqual(result["media_type"], "video/webm")
        self.assertEqual(result["modality"], "video")

    def test_convert_video_block_base64_default_media_type(self) -> None:
        """Test converting video block with base64 source using default media_type"""
        source: Dict[str, Any] = {
            "type": "base64",
            "media_type": "video/mp4",
            "data": "base64data",
        }
        block: VideoBlock = {
            "type": "video",
            "source": source,
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "blob")
        self.assertEqual(result["media_type"], "video/mp4")

    def test_convert_video_block_invalid_source_type(self) -> None:
        """Test converting video block with invalid source type"""
        block: VideoBlock = {
            "type": "video",
            "source": {
                "type": "invalid",
            },
        }
        result = _convert_block_to_part(block)

        self.assertIsNone(result)

    def test_convert_unknown_block_type(self) -> None:
        """Test converting unknown block type"""
        block = {
            "type": "unknown_type",
        }
        result = _convert_block_to_part(block)

        self.assertIsNone(result)

    def test_convert_block_missing_type(self) -> None:
        """Test converting block with missing type field"""
        block = {}
        result = _convert_block_to_part(block)

        self.assertIsNone(result)

    def test_convert_image_block_url_missing_url(self) -> None:
        """Test converting image block with URL source missing url field"""
        block: ImageBlock = {
            "type": "image",
            "source": {
                "type": "url",
            },
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "uri")
        self.assertEqual(result["uri"], "")  # Default empty string

    def test_convert_image_block_base64_missing_fields(self) -> None:
        """Test converting image block with base64 source missing optional fields"""
        block: ImageBlock = {
            "type": "image",
            "source": {
                "type": "base64",
            },
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "blob")
        self.assertEqual(result["content"], "")  # Default empty string
        self.assertEqual(result["media_type"], "image/jpeg")  # Default media type

    def test_convert_audio_block_url_missing_url(self) -> None:
        """Test converting audio block with URL source missing url field"""
        block: AudioBlock = {
            "type": "audio",
            "source": {
                "type": "url",
            },
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "uri")
        self.assertEqual(result["uri"], "")  # Default empty string

    def test_convert_audio_block_base64_missing_fields(self) -> None:
        """Test converting audio block with base64 source missing optional fields"""
        block: AudioBlock = {
            "type": "audio",
            "source": {
                "type": "base64",
            },
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "blob")
        self.assertEqual(result["content"], "")  # Default empty string
        self.assertEqual(result["media_type"], "audio/wav")  # Default media type

    def test_convert_video_block_url_missing_url(self) -> None:
        """Test converting video block with URL source missing url field"""
        block: VideoBlock = {
            "type": "video",
            "source": {
                "type": "url",
            },
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "uri")
        self.assertEqual(result["uri"], "")  # Default empty string

    def test_convert_video_block_base64_missing_fields(self) -> None:
        """Test converting video block with base64 source missing optional fields"""
        block: VideoBlock = {
            "type": "video",
            "source": {
                "type": "base64",
            },
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "blob")
        self.assertEqual(result["content"], "")  # Default empty string
        self.assertEqual(result["media_type"], "video/mp4")  # Default media type

    def test_convert_tool_result_block_empty_output(self) -> None:
        """Test converting tool_result block with empty string output"""
        block: ToolResultBlock = {
            "type": "tool_result",
            "id": "call_123",
            "name": "search",
            "output": "",
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "tool_call_response")
        self.assertEqual(result["response"], "")

    def test_convert_tool_result_block_none_output(self) -> None:
        """Test converting tool_result block with None output"""
        block = {
            "type": "tool_result",
            "id": "call_123",
            "name": "search",
            "output": None,
        }
        result = _convert_block_to_part(block)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "tool_call_response")
        self.assertEqual(result["response"], "None")
