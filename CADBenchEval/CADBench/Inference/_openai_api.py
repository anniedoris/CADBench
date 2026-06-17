from ._base import InferenceEngine
from ..Processing import VLMProcessor, Processor
from openai import OpenAI
from typing import Any, Dict, Optional, Union
import re


class APIInferenceEngine(InferenceEngine):
    def __init__(self,
                 processor: Union[VLMProcessor, Processor],
                 model: str,
                 num_workers: int = 16,
                 api_endpoint: Optional[str] = "http://localhost:8000/v1",
                 api_key: Optional[str] = None,
                 verbose: bool = True,
                 **kwargs: Any
                 ):
        super().__init__(processor, num_workers, verbose)
        if api_key is None:
            api_key = "KEY_NOT_SET"
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.client = OpenAI(api_key=api_key, base_url=api_endpoint)
        self.model = model
        self.api_kwargs = kwargs

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['client']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_endpoint)

    @staticmethod
    def extract_code(raw_response: str) -> str:
        # Try ```python``` first
        match = re.search(r"```python\s*\n(.*?)```", raw_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Try any labeled code block (e.g. ```py, ```cadquery, etc.)
        match = re.search(r"```\w+\s*\n(.*?)```", raw_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Try unlabeled code block
        match = re.search(r"```\s*\n(.*?)```", raw_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw_response

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [APIInferenceEngine._coerce_text(part) for part in value]
            return "\n".join(part for part in parts if part)
        if isinstance(value, dict):
            for key in ("text", "content", "value"):
                if key in value:
                    return APIInferenceEngine._coerce_text(value[key])
            return str(value)

        text_attr = getattr(value, "text", None)
        if text_attr is not None:
            return APIInferenceEngine._coerce_text(text_attr)

        content_attr = getattr(value, "content", None)
        if content_attr is not None and content_attr is not value:
            return APIInferenceEngine._coerce_text(content_attr)

        return str(value)

    @staticmethod
    def _get_message_texts(message: Any) -> Dict[str, str]:
        content = APIInferenceEngine._coerce_text(getattr(message, "content", None))
        reasoning = APIInferenceEngine._coerce_text(
            getattr(message, "reasoning_content", None)
        )

        if reasoning and content:
            raw_response = f"<think>\n{reasoning}\n</think>\n\n{content}"
        elif reasoning:
            raw_response = f"<think>\n{reasoning}\n</think>"
        else:
            raw_response = content

        return {
            "content": content,
            "reasoning": reasoning,
            "raw_response": raw_response,
        }

    def _run_inference(self, input_data: Any) -> Any:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=input_data,
            **self.api_kwargs
        )
        message = response.choices[0].message
        message_texts = self._get_message_texts(message)

        # Prefer the final assistant content for evaluation, but fall back to
        # the full trace when thinking models omit a post-reasoning answer.
        code_source = message_texts["content"] or message_texts["raw_response"]
        code = self.extract_code(code_source)
        return {
            "generated": code,
            "raw_response": message_texts["raw_response"],
            "reasoning_content": message_texts["reasoning"] or None,
        }
