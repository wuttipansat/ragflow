import os
from typing import Any

import requests

from ragflow.generation.prompt import SYSTEM_PROMPT


class OpenRouterGenerator:
    """Generate answers using the OpenRouter API."""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.1,
        max_tokens: int = 500,
        timeout_seconds: int = 120,
    ) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY was not found. "
                "Add it to the .env file."
            )

        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

        self.last_model_used: str | None = None

    def generate(
        self,
        prompt: str,
    ) -> str:
        """Generate an answer from a RAG prompt."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )

            response.raise_for_status()

        except requests.Timeout as error:
            raise RuntimeError(
                "OpenRouter took too long to respond."
            ) from error

        except requests.ConnectionError as error:
            raise RuntimeError(
                "Cannot connect to OpenRouter."
            ) from error

        except requests.HTTPError as error:
            raise RuntimeError(
                f"OpenRouter request failed: "
                f"{response.status_code}\n"
                f"{response.text}"
            ) from error

        response_data = response.json()

        self.last_model_used = response_data.get("model")

        choices = response_data.get("choices", [])

        if not choices:
            raise RuntimeError(
                f"OpenRouter returned no choices: "
                f"{response_data}"
            )

        answer = (
            choices[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not answer:
            raise RuntimeError(
                "OpenRouter returned an empty answer."
            )

        return answer