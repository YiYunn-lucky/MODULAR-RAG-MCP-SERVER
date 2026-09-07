"""DeepSeek Vision LLM implementation.

DeepSeek's vision model (deepseek-v4-flash-vision-exp) uses an OpenAI-compatible
chat completions API, so this provider reuses the OpenAIVisionLLM protocol logic
with DeepSeek's endpoint, model namespace, and DEEPSEEK_API_KEY auth.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from src.libs.llm.openai_vision_llm import OpenAIVisionLLM


class DeepSeekVisionLLMError(RuntimeError):
    """Raised when DeepSeek Vision API call fails."""


class DeepSeekVisionLLM(OpenAIVisionLLM):
    """DeepSeek Vision LLM provider implementation.

    Implements the BaseVisionLLM interface against DeepSeek's OpenAI-compatible
    chat completions endpoint. Images are sent as base64 data URLs.

    Example:
        >>> settings = load_settings('config/settings.yaml')
        >>> vision_llm = DeepSeekVisionLLM(settings)
        >>> image = ImageInput(path="diagram.png")
        >>> response = vision_llm.chat_with_image("Describe this", image)
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DeepSeek Vision LLM provider.

        API key resolution order: explicit param > vision_llm.api_key >
        llm.api_key > DEEPSEEK_API_KEY environment variable.

        Raises:
            ValueError: If no API key can be resolved.
        """
        vision_settings = getattr(settings, "vision_llm", None)

        resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not resolved_key and vision_settings:
            resolved_key = getattr(vision_settings, "api_key", None)
        if not resolved_key:
            resolved_key = getattr(settings.llm, "api_key", None)
        if not resolved_key:
            raise ValueError(
                "DeepSeek API key not provided. Set in settings.yaml "
                "(vision_llm.api_key), DEEPSEEK_API_KEY environment variable, "
                "or pass api_key parameter."
            )

        super().__init__(
            settings=settings,
            api_key=resolved_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            **kwargs,
        )
