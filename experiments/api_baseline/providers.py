"""Provider requests for the shared tool-free evaluator."""
import asyncio
import os
import time
from pathlib import Path

from openai import APITimeoutError

from .evaluation import build_request, evaluate_one, parse_answer, task_instructions

EXAONE_URL = "https://api.exaone.ai/v1"
KEY_NAMES = {"openai": "OPENAI_API_KEY", "exaone": "EXAONE_API_KEY"}
EFFORTS = {"openai": ("low", "medium", "high", "xhigh", "max"), "exaone": ("low", "high", "max")}
REQUEST_TIMEOUTS = (asyncio.TimeoutError, APITimeoutError)


class DeadlineClient:
    """Reuse one async SDK session across synchronous experiment batches."""

    def __init__(self, sdk, timeout):
        self.sdk = sdk
        self.timeout = timeout
        self.loop = asyncio.new_event_loop()

    def run(self, awaitable):
        return self.loop.run_until_complete(awaitable)

    async def request(self, mode, payload):
        send = self.sdk.chat.completions.create if mode == "exaone" else self.sdk.responses.create
        return await asyncio.wait_for(send(**payload), timeout=self.timeout)

    def close(self):
        try:
            self.run(self.sdk.close())
        finally:
            self.loop.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ReflectionLimit(RuntimeError):
    """Reflection exceeded its time or generation budget."""


def validate_config(config):
    provider = config.get("provider", "openai")
    if provider not in KEY_NAMES:
        raise ValueError("provider must be openai or exaone")
    if config.get("reasoning_effort") not in EFFORTS[provider]:
        raise ValueError(f"Invalid reasoning_effort for {provider}: choose from {', '.join(EFFORTS[provider])}")
    if type(config.get("max_output_tokens")) is not int or config["max_output_tokens"] <= 0:
        raise ValueError("max_output_tokens must be a positive integer")
    model = config.get("model")
    if not isinstance(model, str) or not model.strip() or Path(model).name != model or model in (".", ".."):
        raise ValueError("model must be a valid model ID without path separators")
    timeout = config.get("timeout_seconds")
    if not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
        raise ValueError("timeout_seconds must be positive and finite")
    return provider


def create_client(config):
    from openai import AsyncOpenAI, OpenAI

    provider = config.get("provider", "openai")
    key_name = KEY_NAMES[provider]
    if not os.environ.get(key_name, "").strip():
        raise ValueError(f"Set {key_name} in .env or the remote Secret")
    options = {"max_retries": 0, "timeout": config["timeout_seconds"]}
    if "provider" in config:
        url = EXAONE_URL if provider == "exaone" else "https://api.openai.com/v1"
        return DeadlineClient(AsyncOpenAI(api_key=os.environ[key_name], base_url=url, **options),
                              timeout=config["timeout_seconds"])
    return OpenAI(**options)


def exaone_request(config, instructions, text):
    return {
        "model": config["model"],
        "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": text}],
        "max_tokens": config["max_output_tokens"],
        "reasoning_effort": config["reasoning_effort"],
        "stream": False,
    }


def task_request(question, config, prompt):
    if config.get("provider") == "exaone":
        return exaone_request(config, task_instructions(config, prompt), question)
    return build_request(question, config, prompt)


def exaone_response(response):
    raw = response.model_dump(mode="json")
    choices = raw.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    usage = raw.get("usage") or {}
    normalized = {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "input_tokens_details": usage.get("prompt_tokens_details"),
        "output_tokens_details": {
            "reasoning_tokens": usage.get("reasoning_tokens") or
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        },
    }
    finish = choice.get("finish_reason")
    if message.get("tool_calls") or message.get("function_call") or finish in ("tool_calls", "function_call"):
        status = "unexpected_tool_call"
    elif message.get("refusal") or finish == "content_filter":
        status = "refusal"
    elif finish == "length":
        status = "token_limit"
    else:
        status = "completed" if finish == "stop" else "incomplete"
    return raw, message.get("content") or "", normalized, status


def openai_response(response):
    raw = response.model_dump(mode="json")
    status = raw.get("status")
    items = raw.get("output") or []
    if any(item.get("type", "").endswith("_call") for item in items):
        status = "unexpected_tool_call"
    elif (raw.get("incomplete_details") or {}).get("reason") == "max_output_tokens":
        status = "token_limit"
    elif status != "completed":
        status = "incomplete"
    elif any(content.get("type") == "refusal" for item in items for content in item.get("content", [])):
        status = "refusal"
    return raw, response.output_text or "", raw.get("usage"), status


async def evaluate_new_task(client, row, config, prompt):
    start = time.monotonic()
    record = {"index": row["index"], "question": row["question"], "expected_answer": row["answer"]}
    provider = config["provider"]
    try:
        response = await client.request(provider, task_request(row["question"], config, prompt))
        decode = exaone_response if provider == "exaone" else openai_response
        raw, text, usage, status = decode(response)
    except Exception as exc:
        status = "timeout" if isinstance(exc, REQUEST_TIMEOUTS) else "api_error"
        return dict(record, status=status, correct=False, parsed_answer=None, output_text="",
                    error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None),
                    latency_seconds=time.monotonic() - start, usage=None)
    parsed = parse_answer(text)
    if status == "completed":
        status = "format_error" if parsed is None else "correct" if parsed == row["answer"] else "incorrect"
    return dict(record, status=status, correct=status == "correct", parsed_answer=parsed, output_text=text,
                response=raw, usage=usage, latency_seconds=time.monotonic() - start)


def evaluate_task(client, row, config, prompt):
    if "provider" not in config:
        return evaluate_one(client, row, config, prompt)
    return client.run(evaluate_new_task(client, row, config, prompt))


def evaluate_wave(client, rows, config, prompt):
    async def evaluate():
        return await asyncio.gather(*(evaluate_new_task(client, row, config, prompt) for row in rows))

    return client.run(evaluate())


def reflect(client, config, prompt):
    provider = config["provider"]
    if provider == "exaone":
        request = exaone_request(config, "Follow the prompt revision instructions.", prompt)
        decode = exaone_response
    else:
        request = {
            "model": config["model"], "input": prompt, "reasoning": {"effort": config["reasoning_effort"]},
            "max_output_tokens": config["max_output_tokens"], "tools": [], "tool_choice": "none", "store": False,
        }
        decode = openai_response
    try:
        response = client.run(client.request(provider, request))
        _, text, usage, status = decode(response)
    except REQUEST_TIMEOUTS:
        raise ReflectionLimit("Reflection timed out") from None
    except Exception as exc:
        raise RuntimeError(f"Reflection API failed ({type(exc).__name__}, HTTP {getattr(exc, 'status_code', None)})") from None
    if status == "token_limit":
        raise ReflectionLimit("Reflection reached its token limit")
    if status != "completed" or not text or not text.strip():
        raise RuntimeError(f"Reflection returned no complete text (status={status})")
    return text, usage
