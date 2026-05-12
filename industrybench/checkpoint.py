import json
import os
import threading
from typing import Dict, List


class CheckpointManager:
    def __init__(self, path: str):
        self.path = path
        self.state: Dict = {"completed_ids": [], "results": []}
        self.lock = threading.RLock()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.state = json.load(f)

    def save(self):
        with self.lock:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.state, f)

    def mark_completed(self, sample_id: str, result: Dict):
        with self.lock:
            if sample_id not in self.state["completed_ids"]:
                self.state["completed_ids"].append(sample_id)
                self.state["results"].append(result)
            self.save()

    def mark_failed(self, sample_id: str):
        self.save()

    def get_existing_results(self) -> List[Dict]:
        return list(self.state.get("results", []))
