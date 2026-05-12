import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List

from .checkpoint import CheckpointManager


class BatchProcessor:
    def __init__(
        self,
        max_concurrency: int = 5,
        request_interval: float = 0.5,
        max_retries: int = 3,
        retry_interval: float = 5.0,
        checkpoint_file: str = "checkpoint.json",
        enable_checkpoint: bool = True,
    ):
        self.max_concurrency = max_concurrency
        self.request_interval = request_interval
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.checkpoint = CheckpointManager(checkpoint_file) if enable_checkpoint else None
        self.lock = threading.Lock()
        self.completed = 0
        self.failed = 0
        self.total = 0

    def process(self, items: List[Dict], process_func: Callable, item_id_key: str = "id") -> List[Dict]:
        self.total = len(items)
        if self.checkpoint:
            self.checkpoint.state["total_samples"] = self.total

        if self.checkpoint:
            completed_ids = set(self.checkpoint.state["completed_ids"])
            pending = [item for item in items if str(item[item_id_key]) not in completed_ids]
            results = self.checkpoint.get_existing_results()
            self.completed = len(completed_ids)
        else:
            pending = items
            results = []

        print(f"\n{'='*60}")
        print(f"Task: {self.total} samples, {len(pending)} pending, concurrency={self.max_concurrency}")
        print(f"{'='*60}\n")

        if not pending:
            print("All samples already completed (resumed from checkpoint)")
            return results

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            future_to_item = {}
            for item in pending:
                future = executor.submit(self._process_with_retry, item, process_func, item_id_key)
                future_to_item[future] = item
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    with self.lock:
                        results.append(result)
                        self.completed += 1
                        if self.checkpoint:
                            self.checkpoint.mark_completed(str(item[item_id_key]), result)
                    self._print_progress()
                except Exception as e:
                    with self.lock:
                        self.failed += 1
                        if self.checkpoint:
                            self.checkpoint.mark_failed(str(item[item_id_key]))
                    self._print_progress()
                    print(f"\n  [ERROR] {item[item_id_key]}: {e}")

        print(f"\n{'='*60}")
        print(f"Done: {self.completed} success, {self.failed} failed")
        print(f"{'='*60}\n")
        return results

    def _process_with_retry(self, item, process_func, item_id_key):
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = process_func(item)
                if not result.get("model_response", "").strip():
                    raise RuntimeError("Empty model response")
                if self.request_interval > 0:
                    time.sleep(self.request_interval)
                return result
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = self.retry_interval * attempt
                    print(f"\n  [RETRY] {item[item_id_key]} attempt {attempt}: {e}, wait {wait}s")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_error}") from last_error

    def _print_progress(self):
        pct = round(self.completed / self.total * 100, 1) if self.total > 0 else 0
        print(f"  [Progress] {self.completed}/{self.total} ({pct}%) | Failed: {self.failed}", end="\r")
