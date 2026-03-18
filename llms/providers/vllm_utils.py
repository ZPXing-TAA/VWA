"""Tools to generate from a vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import os
import random
import time
from typing import Any

import openai
from openai import OpenAI


VLLM_API_KEY_ENV = "VLLM_API_KEY"
DEFAULT_VLLM_API_KEY = "EMPTY"


def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple[Any, ...] = (
        openai.RateLimitError,
        openai.BadRequestError,
        openai.InternalServerError,
    ),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)
            except Exception as e:
                raise e

    return wrapper


def _get_vllm_client(model_endpoint: str) -> OpenAI:
    if not model_endpoint:
        raise ValueError(
            "model_endpoint must be set when using the vLLM provider."
        )

    api_key = os.environ.get(VLLM_API_KEY_ENV, DEFAULT_VLLM_API_KEY)
    return OpenAI(base_url=model_endpoint, api_key=api_key)


@retry_with_exponential_backoff
def generate_from_vllm_completion(
    prompt: str,
    model: str,
    model_endpoint: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_token: str | None = None,
) -> str:
    client = _get_vllm_client(model_endpoint)
    stop = [stop_token] if stop_token else None
    response = client.completions.create(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
    )
    return response.choices[0].text


@retry_with_exponential_backoff
def generate_from_vllm_chat_completion(
    messages: list[dict[str, Any]],
    model: str,
    model_endpoint: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    num_outputs: int = 1,
) -> str | list[str]:
    client = _get_vllm_client(model_endpoint)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=num_outputs,
    )
    if num_outputs > 1:
        return [choice.message.content for choice in response.choices]
    return response.choices[0].message.content
