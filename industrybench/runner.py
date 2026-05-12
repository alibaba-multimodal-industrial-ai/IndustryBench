import csv
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from .batch import BatchProcessor
from .data import QASample
from .judge import LLMJudge
from .openai_client import call_llm
from .retry import retry_call
from .safety import SafetyReviewer


def run_single_model(
    model_name: str,
    samples: List[QASample],
    api_key: str,
    api_base: str,
    judge_model: str,
    temperature: float,
    max_tokens: Optional[int],
    prompt_template: str,
    enable_safety_review: bool,
    concurrency: int,
    checkpoint_file: str,
    output_dir: str,
) -> List[Dict]:
    judge = LLMJudge(judge_model, call_llm, api_key=api_key, api_base=api_base)
    safety_reviewer = (
        SafetyReviewer(judge_model, call_llm, api_key=api_key, api_base=api_base) if enable_safety_review else None
    )

    def _format_prompt(template: str, question: str) -> str:
        return template.replace("${question}", question)

    def process_sample(sample_dict: Dict) -> Dict:
        sample = QASample(**sample_dict) if isinstance(sample_dict, dict) else sample_dict
        prompt_text = _format_prompt(prompt_template, sample.question)
        start_time = time.time()
        messages = [{"role": "user", "content": prompt_text}]

        model_result = call_llm(
            messages=messages,
            model=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response_time = time.time() - start_time

        if not model_result["success"] or not model_result.get("content", "").strip():
            return {
                "id": sample.id,
                "question": sample.question,
                "answer": sample.ground_truth,
                "model_response": "",
                "model_name": model_name,
                "prompt_used": prompt_text,
                "success": False,
                "error_message": model_result.get("error", "Unknown error") or "Empty response",
                "score": -1,
                "judge_reason": "",
                "safety_violation": None,
                "safety_review_reason": "",
                "final_score": -1,
                "domain": sample.domain,
                "difficulty": sample.difficulty,
                "capability": sample.capability,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "response_time": round(response_time, 2),
            }

        model_response = model_result["content"]
        usage = model_result.get("usage") or {}

        judge_result = retry_call(
            lambda: judge.evaluate(
                question=sample.question,
                ground_truth=sample.ground_truth,
                model_response=model_response,
            ),
            label=f"judge {sample.id}",
        )

        if not judge_result.get("success"):
            return {
                "id": sample.id,
                "question": sample.question,
                "answer": sample.ground_truth,
                "model_response": model_response,
                "model_name": model_name,
                "prompt_used": prompt_text,
                "success": False,
                "error_message": f"Judge failed: {judge_result.get('error', '')}",
                "score": -1,
                "judge_reason": "",
                "safety_violation": None,
                "safety_review_reason": "",
                "final_score": -1,
                "domain": sample.domain,
                "difficulty": sample.difficulty,
                "capability": sample.capability,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": 0,
                "total_tokens": 0,
                "response_time": round(response_time, 2),
            }

        safety_result: Dict[str, Any] = {"safety_violation": None, "reason": "", "success": False, "error": None}
        if safety_reviewer:
            safety_result = retry_call(
                lambda: safety_reviewer.review(
                    question=sample.question,
                    ground_truth=sample.ground_truth,
                    knowledge_text=sample.knowledge_text,
                    model_response=model_response,
                ),
                label=f"safety {sample.id}",
            )
            if not safety_result.get("success"):
                safety_result = {
                    "safety_violation": None,
                    "reason": safety_result.get("error", "Safety review failed"),
                }

        raw_score = judge_result["score"]
        safety_violation = safety_result.get("safety_violation")
        final_score = 0.0 if safety_violation is True else raw_score

        return {
            "id": sample.id,
            "question": sample.question,
            "answer": sample.ground_truth,
            "model_response": model_response,
            "model_name": model_name,
            "prompt_used": prompt_text,
            "success": True,
            "error_message": None,
            "score": raw_score,
            "judge_reason": judge_result["reason"],
            "safety_violation": safety_violation,
            "safety_review_reason": safety_result.get("reason", ""),
            "final_score": final_score,
            "domain": sample.domain,
            "difficulty": sample.difficulty,
            "capability": sample.capability,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "response_time": round(response_time, 2),
        }

    sample_dicts = [
        {
            "id": s.id,
            "question": s.question,
            "ground_truth": s.ground_truth,
            "domain": s.domain,
            "difficulty": s.difficulty,
            "capability": s.capability,
            "knowledge_text": s.knowledge_text,
            "metadata": s.metadata,
        }
        for s in samples
    ]

    batch = BatchProcessor(max_concurrency=concurrency, checkpoint_file=checkpoint_file)
    results = batch.process(items=sample_dicts, process_func=process_sample, item_id_key="id")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace("/", "_").replace(" ", "_")

    csv_path = os.path.join(output_dir, f"{safe_name}_results_{timestamp}.csv")
    if results:
        fieldnames = list(results[0].keys())
        priority = [
            "id",
            "question",
            "answer",
            "model_response",
            "model_name",
            "success",
            "score",
            "final_score",
            "safety_violation",
            "safety_review_reason",
            "judge_reason",
        ]
        ordered = [f for f in priority if f in fieldnames] + [f for f in fieldnames if f not in priority]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=ordered)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\n  Results saved to: {csv_path}")

    print_summary(results, model_name)
    return results


def print_summary(results: List[Dict], model_name: str = ""):
    if not results:
        return
    total = len(results)
    success_count = sum(1 for r in results if r.get("success"))
    scores = [r.get("score", -1) for r in results if r.get("success") and r.get("score", -1) >= 0]
    final_scores = [r.get("final_score", -1) for r in results if r.get("success") and r.get("final_score", -1) >= 0]
    safety_reviewed = [r for r in results if r.get("safety_violation") is not None]
    safety_violations = sum(1 for r in safety_reviewed if r.get("safety_violation") is True)
    times = [r.get("response_time", 0) for r in results if r.get("success") and r.get("response_time", 0) > 0]

    print(f"\n{'='*60}")
    if model_name:
        print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"  Total:      {total}")
    print(f"  Success:    {success_count}")
    print(f"  Failed:     {total - success_count}")
    print(f"  Success %:  {round(success_count / total * 100, 1) if total > 0 else 0}%")
    if scores:
        print(f"\n  --- Raw Score (0-3) ---")
        print(f"  Mean:       {round(sum(scores) / len(scores), 2)}")
        print(f"  Median:     {round(sorted(scores)[len(scores) // 2], 2)}")
        print(f"  Min/Max:    {min(scores)} / {max(scores)}")
        dist = {}
        for s in scores:
            bucket = f"{int(s)}pt"
            dist[bucket] = dist.get(bucket, 0) + 1
        print(f"  Distribution:")
        for b, c in sorted(dist.items()):
            print(f"    {b}: {c}")
    if final_scores:
        print(f"\n  --- Final Score (after safety review) ---")
        print(f"  Mean:       {round(sum(final_scores) / len(final_scores), 2)}")
        print(f"  Median:     {round(sorted(final_scores)[len(final_scores) // 2], 2)}")
    if safety_reviewed:
        print(f"\n  --- Safety Review ---")
        print(f"  Violations: {safety_violations} ({round(safety_violations / len(safety_reviewed) * 100, 1)}%)")
    if times:
        print(f"\n  Time:       {round(sum(times), 1)}s total, {round(sum(times) / len(times), 2)}s avg")
    print(f"{'='*60}\n")
