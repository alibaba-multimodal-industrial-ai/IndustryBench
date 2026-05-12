import json
from typing import Any, Callable, Dict


JUDGE_PROMPT_TEMPLATE = """You are a professional evaluator for LLM responses. Score the model's answer against the ground truth.

Scrubric (0-3 scale):
- 3: Model answer is essentially consistent with the ground truth, with matching reasoning and logic.
- 2: Model answer is largely correct but differs in reasoning or justification.
- 1: Model answer is partially correct or has plausible reasoning but incorrect conclusion.
- 0: Model answer is entirely incorrect.

Output strict JSON with fields "score" (integer 0-3) and "reason" (string).

Question: {question}
Ground truth: {ground_truth}
Model answer: {model_response}"""


class LLMJudge:
    def __init__(self, judge_model: str, call_llm_func: Callable, api_key: str, api_base: str = "https://api.openai.com/v1"):
        self.judge_model = judge_model
        self.call_llm = call_llm_func
        self.api_key = api_key
        self.api_base = api_base

    def evaluate(self, question: str, ground_truth: str, model_response: str) -> Dict[str, Any]:
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            model_response=model_response,
        )
        messages = [
            {"role": "system", "content": "You are a professional evaluator. Output only valid JSON."},
            {"role": "user", "content": judge_prompt},
        ]
        result = self.call_llm(
            messages=messages,
            model=self.judge_model,
            api_key=self.api_key,
            api_base=self.api_base,
            temperature=0.0,
            max_tokens=500,
        )
        if not result["success"]:
            return {"score": -1, "reason": "", "success": False, "error": result.get("error", "Unknown error")}

        try:
            content = result["content"].strip()
            json_str = content
            if "```" in content:
                for part in content.split("```"):
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    try:
                        json.loads(part)
                        json_str = part
                        break
                    except json.JSONDecodeError:
                        continue
            data = json.loads(json_str)
            return {
                "score": float(data.get("score", -1)),
                "reason": data.get("reason", ""),
                "success": True,
                "error": None,
            }
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            return {
                "score": -1,
                "reason": "",
                "success": False,
                "error": f"Parse error: {e}. Raw: {result['content'][:200]}",
            }
