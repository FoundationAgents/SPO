import asyncio
import re
from enum import Enum
from typing import Any, List, Optional
from llm.async_llm import AsyncLLM, LLMsConfig
from loguru import logger


class RequestType(Enum):
    OPTIMIZE = "optimize"
    EVALUATE = "evaluate"
    EXECUTE = "execute"


class SPO_LLM:
    _instance: Optional["SPO_LLM"] = None

    def __init__(
        self,
        optimize_kwargs: Optional[dict] = None,
        evaluate_kwargs: Optional[dict] = None,
        execute_kwargs: Optional[dict] = None,
    ) -> None:
        self.evaluate_llm = AsyncLLM(config=self._load_llm_config(evaluate_kwargs))
        self.optimize_llm = AsyncLLM(config=self._load_llm_config(optimize_kwargs))
        self.execute_llm = AsyncLLM(config=self._load_llm_config(execute_kwargs))

    def _load_llm_config(self, kwargs: dict) -> Any:
        model = kwargs.get("model")
        if not model:
            raise ValueError("'model' parameter is required")

        try:
            model_config = LLMsConfig.default().get("gpt-4o-mini")
            if model_config is None:
                raise ValueError(f"Model gpt-4o-mini not found in configuration")

            config = model_config

            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            return config

        except AttributeError:
            raise ValueError(f"Model '{model}' not found in configuration")
        except Exception as e:
            raise ValueError(f"Error loading configuration for model '{model}': {str(e)}")

    async def responser(self, request_type: RequestType, messages: List[dict]) -> str:
        llm_mapping = {
            RequestType.OPTIMIZE: self.optimize_llm,
            RequestType.EVALUATE: self.evaluate_llm,
            RequestType.EXECUTE: self.execute_llm,
        }

        llm = llm_mapping.get(request_type)
        if not llm:
            raise ValueError(f"Invalid request type. Valid types: {', '.join([t.value for t in RequestType])}")

        response = await llm(messages)
        return response

    @classmethod
    def initialize(cls, optimize_kwargs: dict, evaluate_kwargs: dict, execute_kwargs: dict) -> None:
        """Initialize the global instance"""
        cls._instance = cls(optimize_kwargs, evaluate_kwargs, execute_kwargs)

    @classmethod
    def get_instance(cls) -> "SPO_LLM":
        """Get the global instance"""
        if cls._instance is None:
            raise RuntimeError("SPO_LLM not initialized. Call initialize() first.")
        return cls._instance


def extract_content(xml_string: str, tag: str) -> Optional[str]:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, xml_string, re.DOTALL)
    return match.group(1).strip() if match else None


