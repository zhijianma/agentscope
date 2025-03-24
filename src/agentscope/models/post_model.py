# -*- coding: utf-8 -*-
"""Model wrapper for post-based inference apis."""
import json
import time
from abc import ABC
from typing import Any, Union, List, Optional

import requests
from loguru import logger

from .model import ModelWrapperBase, ModelResponse
from ..constants import _DEFAULT_MAX_RETRIES
from ..constants import _DEFAULT_MESSAGES_KEY
from ..constants import _DEFAULT_RETRY_INTERVAL
from ..formatters import OpenAIFormatter, GeminiFormatter, CommonFormatter
from ..message import Msg


class PostAPIModelWrapperBase(ModelWrapperBase, ABC):
    """The base model wrapper for the model deployed on the POST API."""

    model_type: str

    def __init__(
        self,
        config_name: str,
        api_url: str,
        model_name: Optional[str] = None,
        headers: dict = None,
        max_length: int = 2048,
        timeout: int = 30,
        json_args: dict = None,
        post_args: dict = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        messages_key: str = _DEFAULT_MESSAGES_KEY,
        retry_interval: int = _DEFAULT_RETRY_INTERVAL,
        **kwargs: Any,
    ) -> None:
        """Initialize the model wrapper.

        Args:
            config_name (`str`):
                The id of the model.
            api_url (`str`):
                The url of the post request api.
            model_name (`str`):
                The name of the model. If `None`, the model name will be
                extracted from the `json_args`.
            headers (`dict`, defaults to `None`):
                The headers of the api. Defaults to None.
            max_length (`int`, defaults to `2048`):
                The maximum length of the model.
            timeout (`int`, defaults to `30`):
                The timeout of the api. Defaults to 30.
            json_args (`dict`, defaults to `None`):
                The json arguments of the api. Defaults to None.
            post_args (`dict`, defaults to `None`):
                The post arguments of the api. Defaults to None.
            max_retries (`int`, defaults to `3`):
                The maximum number of retries when the `parse_func` raise an
                exception.
            messages_key (`str`, defaults to `inputs`):
                The key of the input messages in the json argument.
            retry_interval (`int`, defaults to `1`):
                The interval between retries when a request fails.

        Note:
            When an object of `PostApiModelWrapper` is called, the arguments
            will of post requests will be used as follows:

            .. code-block:: python

                request.post(
                    url=api_url,
                    headers=headers,
                    json={
                        messages_key: messages,
                        **json_args
                    },
                    **post_args
                )
        """
        if model_name is None:
            if json_args is not None:
                model_name = json_args.get(
                    "model",
                    json_args.get("model_name", None),
                )
            else:
                model_name = None

        super().__init__(config_name=config_name, model_name=model_name)

        self.api_url = api_url
        self.headers = headers
        self.max_length = max_length
        self.timeout = timeout
        self.json_args = json_args or {}
        self.post_args = post_args or {}
        self.max_retries = max_retries
        self.messages_key = messages_key
        self.retry_interval = retry_interval

    def _parse_response(self, response: dict) -> ModelResponse:
        """Parse the response json data into ModelResponse"""
        return ModelResponse(raw=response)

    def __call__(self, input_: str, **kwargs: Any) -> ModelResponse:
        """Calling the model with requests.post.

        Args:
            input_ (`str`):
                The input string to the model.

        Returns:
            `dict`: A dictionary that contains the response of the model and
            related
            information (e.g. cost, time, the number of tokens, etc.).

        Note:
            `parse_func`, `fault_handler` and `max_retries` are reserved for
            `_response_parse_decorator` to parse and check the response
            generated by model wrapper. Their usages are listed as follows:
                - `parse_func` is a callable function used to parse and check
                the response generated by the model, which takes the response
                as input.
                - `max_retries` is the maximum number of retries when the
                `parse_func` raise an exception.
                - `fault_handler` is a callable function which is called
                when the response generated by the model is invalid after
                `max_retries` retries.
        """
        # step1: prepare keyword arguments
        post_args = {**self.post_args, **kwargs}

        request_kwargs = {
            "url": self.api_url,
            "json": {self.messages_key: input_, **self.json_args},
            "headers": self.headers or {},
            **post_args,
        }

        # step2: prepare post requests
        for i in range(1, self.max_retries + 1):
            response = requests.post(**request_kwargs)

            if response.status_code == requests.codes.ok:
                break

            if i < self.max_retries:
                logger.warning(
                    f"Failed to call the model with "
                    f"requests.codes == {response.status_code}, retry "
                    f"{i + 1}/{self.max_retries} times",
                )
                time.sleep(i * self.retry_interval)

        # step3: record model invocation
        # record the model api invocation, which will be skipped if
        # `FileManager.save_api_invocation` is `False`
        try:
            response_json = response.json()
        except requests.exceptions.JSONDecodeError as e:
            raise RuntimeError(
                f"Fail to serialize the response to json: \n{str(response)}",
            ) from e

        self._save_model_invocation(
            arguments=request_kwargs,
            response=response_json,
        )

        # step4: parse the response
        if response.status_code == requests.codes.ok:
            return self._parse_response(response_json)
        else:
            logger.error(json.dumps(request_kwargs, indent=4))
            raise RuntimeError(
                f"Failed to call the model with {response.json()}",
            )


class PostAPIChatWrapper(PostAPIModelWrapperBase):
    """A post api model wrapper compatible with openai chat, e.g., vLLM,
    FastChat."""

    model_type: str = "post_api_chat"

    def _parse_response(self, response: dict) -> ModelResponse:
        return ModelResponse(
            text=response["data"]["response"]["choices"][0]["message"][
                "content"
            ],
        )

    def format(
        self,
        *args: Union[Msg, list[Msg], None],
        multi_agent_mode: bool = True,
    ) -> Union[List[dict]]:
        """Format the input messages into a list of dict according to the model
        name. For example, if the model name is prefixed with "gpt-", the
        input messages will be formatted for OpenAI models.

        Args:
            args (`Union[Msg, list[Msg], None]`):
                The input arguments to be formatted, where each argument
                should be a `Msg` object, or a list of `Msg` objects. The
                `None` input will be ignored.
            multi_agent_mode (`bool`, defaults to `True`):
                Formatting the messages in multi-agent mode or not. If false,
                the messages will be formatted in chat mode, where only a user
                and an assistant roles are involved.

        Returns:
            `Union[List[dict]]`:
                The formatted messages.
        """
        # Format according to the potential model field in the json_args
        model_name = self.json_args.get(
            "model",
            self.json_args.get("model_name", None),
        )

        # OpenAI
        if OpenAIFormatter.is_supported_model(model_name or ""):
            return OpenAIFormatter.format_multi_agent(*args)

        # Gemini
        if GeminiFormatter.is_supported_model(model_name or ""):
            return GeminiFormatter.format_multi_agent(*args)

        # Include DashScope, ZhipuAI, Ollama, the other models supported by
        # litellm and unknown models
        else:
            return CommonFormatter.format_multi_agent(*args)


class PostAPIDALLEWrapper(PostAPIModelWrapperBase):
    """A post api model wrapper compatible with openai dall_e"""

    model_type: str = "post_api_dall_e"

    def _parse_response(self, response: dict) -> ModelResponse:
        if "data" not in response["data"]["response"]:
            if "error" in response["data"]["response"]:
                error_msg = response["data"]["response"]["error"]["message"]
            else:
                error_msg = response["data"]["response"]
            logger.error(f"Error in API call:\n{error_msg}")
            raise ValueError(f"Error in API call:\n{error_msg}")
        urls = [img["url"] for img in response["data"]["response"]["data"]]
        return ModelResponse(image_urls=urls)


class PostAPIEmbeddingWrapper(PostAPIModelWrapperBase):
    """
    A post api model wrapper for embedding model
    """

    model_type: str = "post_api_embedding"

    def _parse_response(self, response: dict) -> ModelResponse:
        """
        Parse the response json data into ModelResponse with embedding.
        Args:
            response (`dict`):
            The response obtained from the API. This parsing assume the
            structure of the response is the same as OpenAI's as following:
        {
          "object": "list",
          "data": [
            {
              "object": "embedding",
              "embedding": [
                0.0023064255,
                -0.009327292,
                .... (1536 floats total for ada-002)
                -0.0028842222,
              ],
              "index": 0
            }
          ],
          "model": "text-embedding-ada-002",
          "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8
          }
        }
        """
        if (
            "data" not in response
            or len(response["data"]) < 1
            or "embedding" not in response["data"][0]
        ):
            error_msg = json.dumps(response, ensure_ascii=False, indent=2)
            logger.error(f"Error in embedding API call:\n{error_msg}")
            raise ValueError(f"Error in embedding API call:\n{error_msg}")
        embeddings = [data["embedding"] for data in response["data"]]
        return ModelResponse(
            embedding=embeddings,
            raw=response,
        )
