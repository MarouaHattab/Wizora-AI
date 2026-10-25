from typing import ClassVar

from .base import BaseLLMProvider


class NvidiaNimProvider(BaseLLMProvider):
    provider_name: ClassVar[str] = "NVIDIA_NIM"
    base_url: str = "https://integrate.api.nvidia.com/v1"
