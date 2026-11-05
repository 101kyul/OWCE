"""Model calls through inspect-ai."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional

from inspect_ai.model import GenerateConfig, get_model

REQUEST_TIMEOUT_S = 240   # per-request HTTP timeout, overridable with model_cfg["timeout"]
BATCH_TIMEOUT_MARGIN_S = 60   # added to len(prompts) * request timeout for the batch limit


def _maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def generate_with_inspect(model_name: str, model_cfg: Dict[str, Any], prompts: List[str]) -> List[str]:
    """Generate one completion per prompt with ``model_name``. A failed or
    timed-out call yields an empty string and is reported."""
    config = GenerateConfig(
        max_tokens=_maybe_int(model_cfg.get("max_tokens")),
        temperature=_maybe_float(model_cfg.get("temperature")),
        timeout=_maybe_int(model_cfg.get("timeout")) or REQUEST_TIMEOUT_S,
    )

    async def _run() -> List[str]:
        async with get_model(model_name, config=config) as model:
            outputs: List[str] = []
            total = len(prompts)
            n_error = 0
            n_blank = 0
            for prompt in prompts:
                try:
                    result = await model.generate(prompt, config=config)
                    text = result.completion or ""
                except Exception as e:
                    print(f"      [error] {model_name} generate error: {e}")
                    text = ""
                    n_error += 1
                if not text.strip():
                    n_blank += 1
                outputs.append(text)
            if n_error or n_blank:
                print(f"      [blank] {model_name}: {n_error}/{total} calls failed, "
                      f"{n_blank}/{total} returned nothing.")
            return outputs

    # Own thread and event loop, so the whole batch can be given a hard timeout.
    def _run_in_thread() -> List[str]:
        return asyncio.run(_run())

    # The prompts run one after another, so the batch limit is the per-request
    # timeout times the batch size. The executor is not used as a context
    # manager so that a hung worker cannot block the caller.
    request_timeout = _maybe_int(model_cfg.get("timeout")) or REQUEST_TIMEOUT_S
    batch_timeout = request_timeout * max(1, len(prompts)) + BATCH_TIMEOUT_MARGIN_S
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_run_in_thread)
    try:
        return future.result(timeout=batch_timeout)
    except concurrent.futures.TimeoutError:
        print(f"      [timeout] {model_name} batch timeout after "
              f"{batch_timeout}s; {len(prompts)} prompts returned nothing.")
        return [""] * len(prompts)
    except Exception as e:
        print(f"      [error] {model_name} thread error: {e}; "
              f"{len(prompts)} prompts returned nothing.")
        return [""] * len(prompts)
    finally:
        executor.shutdown(wait=False)
