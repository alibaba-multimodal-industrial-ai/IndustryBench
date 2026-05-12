import json
from typing import Any, Dict, List, Optional


def call_llm(
    messages: List[Dict[str, str]],
    model: str,
    api_key: str,
    api_base: str = "https://api.openai.com/v1",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Call LLM via OpenAI-compatible API (streaming)."""
    import requests

    if not api_key:
        return {"success": False, "content": None, "error": "API key is required", "usage": None, "model": model}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    url = api_base.rstrip("/") + "/chat/completions"
    try:
        resp = requests.post(url=url, headers=headers, json=payload, timeout=timeout, stream=True)
        if resp.status_code != 200:
            return {
                "success": False,
                "content": None,
                "error": f"HTTP {resp.status_code}: {resp.text[:500]}",
                "usage": None,
                "model": model,
            }

        full_content = ""
        usage_info = None
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8").strip()
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            full_content += delta["content"]
                    if "usage" in chunk and chunk["usage"]:
                        u = chunk["usage"]
                        usage_info = {
                            "prompt_tokens": u.get("prompt_tokens", 0),
                            "completion_tokens": u.get("completion_tokens", 0),
                            "total_tokens": u.get("total_tokens", 0),
                        }
                except json.JSONDecodeError:
                    continue

        return {"success": True, "content": full_content.strip(), "error": None, "usage": usage_info, "model": model}
    except requests.RequestException as e:
        return {"success": False, "content": None, "error": f"Request failed: {str(e)}", "usage": None, "model": model}
