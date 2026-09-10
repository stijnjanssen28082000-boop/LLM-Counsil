"""Core orchestration for the LLM Council: a 3-stage multi-model deliberation.

Stage 1 — every council model answers the question independently.
Stage 2 — each model ranks the (anonymized) answers of the others.
Stage 3 — a "chairman" model synthesizes a final answer from stages 1 & 2.

All model calls go through OpenRouter's OpenAI-compatible chat completions
endpoint, so any model available there can be used.
"""

from __future__ import annotations

import re
import string
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 120


def query_model(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Call one model on OpenRouter and normalize the result.

    Returns a dict with either 'content' or 'error'/'error_message'.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/stijnjanssen28082000-boop/LLM-Counsil",
        "X-Title": "LLM Council",
    }
    payload = {"model": model, "messages": messages, "temperature": temperature}

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        return {"error": True, "error_message": f"Request failed: {exc}"}

    if resp.status_code != 200:
        detail = resp.text[:300]
        return {"error": True, "error_message": f"HTTP {resp.status_code}: {detail}"}

    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return {"error": True, "error_message": "Empty response from model"}

    content = (choices[0].get("message") or {}).get("content")
    if not content:
        return {"error": True, "error_message": "Model returned no content"}

    usage = data.get("usage") or {}
    return {"content": content, "usage": usage}


def _query_models_parallel(
    api_key: str,
    models: List[str],
    messages: List[Dict[str, str]],
    temperature: float,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(models))) as pool:
        futures = {
            pool.submit(query_model, api_key, model, messages, temperature): model
            for model in models
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                results[model] = future.result()
            except Exception as exc:  # defensive: a worker must never crash the run
                results[model] = {"error": True, "error_message": str(exc)}
    return results


def stage1_collect_responses(
    api_key: str,
    models: List[str],
    user_query: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
) -> List[Dict[str, Any]]:
    """Every council model answers the user's question independently."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_query})

    responses = _query_models_parallel(api_key, models, messages, temperature)

    results = []
    for model in models:
        response = responses.get(model) or {}
        if response.get("error"):
            results.append({
                "model": model,
                "error": True,
                "error_message": response.get("error_message", "Unknown error"),
            })
        else:
            results.append({"model": model, "response": response.get("content", "")})
    return results


LABELS = list(string.ascii_uppercase)


def stage2_collect_rankings(
    api_key: str,
    models: List[str],
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    temperature: float = 0.2,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Each model ranks the anonymized Stage 1 answers."""
    valid = [r for r in stage1_results if r.get("response") and r["response"].strip()]
    if len(valid) < 2:
        return [], {}

    label_to_model = {f"Response {LABELS[i]}": r["model"] for i, r in enumerate(valid)}

    responses_block = "\n\n".join(
        f"Response {LABELS[i]}:\n{r['response']}" for i, r in enumerate(valid)
    )
    ranking_prompt = f"""You are evaluating anonymized answers to this question:

Question: {user_query}

{responses_block}

Evaluate each response for accuracy, clarity and completeness. Then provide your
ranking from best to worst, one label per line, in this exact format at the end:

FINAL RANKING:
1. Response X
2. Response Y
..."""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Only models that succeeded in Stage 1 are asked to rank.
    successful_models = [r["model"] for r in valid]
    ranking_models = [m for m in models if m in successful_models] or successful_models

    responses = _query_models_parallel(api_key, ranking_models, messages, temperature)

    stage2_results = []
    for model in ranking_models:
        response = responses.get(model) or {}
        if response.get("error") or not response.get("content"):
            stage2_results.append({
                "model": model,
                "error": True,
                "error_message": response.get("error_message", "Empty response"),
            })
        else:
            text = response["content"]
            stage2_results.append({
                "model": model,
                "ranking": text,
                "parsed_ranking": parse_ranking_from_text(text),
            })

    return stage2_results, label_to_model


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """Extract the ordered list of 'Response X' labels from a FINAL RANKING section."""
    parts = re.split(r"FINAL RANKING\s*:", ranking_text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) < 2:
        return []
    section = parts[1]
    numbered = re.findall(r"\d+\.\s*Response [A-Z]", section)
    if numbered:
        labels = [re.search(r"Response [A-Z]", m).group() for m in numbered]
    else:
        labels = re.findall(r"Response [A-Z]", section)
    return list(dict.fromkeys(labels))


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Average each model's rank position across all peer rankings."""
    positions = defaultdict(list)
    for ranking in stage2_results:
        if ranking.get("error"):
            continue
        for pos, label in enumerate(ranking.get("parsed_ranking", []), start=1):
            model = label_to_model.get(label)
            if model:
                positions[model].append(pos)

    aggregate = [
        {"model": model, "average_rank": round(sum(p) / len(p), 2), "rankings_count": len(p)}
        for model, p in positions.items()
        if p
    ]
    aggregate.sort(key=lambda x: x["average_rank"])
    return aggregate


def stage3_synthesize_final(
    api_key: str,
    chairman_model: str,
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    temperature: float = 0.5,
) -> Dict[str, Any]:
    """The chairman model reads all answers and rankings, and writes the final answer."""
    valid1 = [r for r in stage1_results if r.get("response")]
    if not valid1:
        return {
            "model": chairman_model,
            "response": "Error: no council model produced a usable answer.",
            "error": True,
        }

    answers_block = "\n\n".join(f"{r['model']}:\n{r['response']}" for r in valid1)

    rankings_block = ""
    valid2 = [r for r in stage2_results if r.get("ranking")]
    if valid2:
        rankings_block = "\n\n".join(f"{r['model']} ranking:\n{r['ranking']}" for r in valid2)
    else:
        rankings_block = "(No peer rankings available.)"

    chairman_prompt = f"""You are the chairman of an LLM council. Council members were
independently asked the question below, and then ranked each other's anonymized answers.

Original question: {user_query}

STAGE 1 — Council answers:
{answers_block}

STAGE 2 — Peer rankings:
{rankings_block}

Write the best possible final answer to the original question, drawing on the
strongest points from the council's answers and taking the peer rankings into
account. Answer directly; do not describe the council process."""

    messages = [{"role": "user", "content": chairman_prompt}]
    response = query_model(api_key, chairman_model, messages, temperature)

    if response.get("error"):
        return {
            "model": chairman_model,
            "response": f"Error: chairman model failed ({response.get('error_message')}).",
            "error": True,
        }

    return {"model": chairman_model, "response": response.get("content", "")}


def run_council(
    api_key: str,
    models: List[str],
    chairman_model: str,
    user_query: str,
    execution_mode: str = "full",
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full 3-stage pipeline (or a shorter mode) and return every stage's output."""
    stage1_results = stage1_collect_responses(api_key, models, user_query, system_prompt)

    result: Dict[str, Any] = {
        "execution_mode": execution_mode,
        "stage1": stage1_results,
        "stage2": None,
        "stage3": None,
        "aggregate_rankings": None,
    }

    if execution_mode == "chat_only":
        return result

    stage2_results, label_to_model = stage2_collect_rankings(
        api_key, models, user_query, stage1_results
    )
    result["stage2"] = stage2_results
    result["aggregate_rankings"] = calculate_aggregate_rankings(stage2_results, label_to_model)

    if execution_mode == "chat_ranking":
        return result

    result["stage3"] = stage3_synthesize_final(
        api_key, chairman_model, user_query, stage1_results, stage2_results
    )
    return result
