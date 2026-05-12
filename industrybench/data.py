import csv
from dataclasses import dataclass, field
from typing import Dict, List

from .languages import LANGUAGE_FIELDS, LANGUAGE_NAMES


@dataclass
class QASample:
    id: str
    question: str
    ground_truth: str
    domain: str = ""
    difficulty: str = ""
    capability: str = ""
    knowledge_text: str = ""
    metadata: dict = field(default_factory=dict)


def load_dataset(file_path: str) -> List[Dict]:
    samples = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = {k.strip(): (v.strip() if v else "") for k, v in row.items()}
            samples.append(sample)
    return samples


def extract_language_samples(data: List[Dict], language: str) -> List[QASample]:
    fields = LANGUAGE_FIELDS[language]
    question_field = fields["question"]
    answer_field = fields["answer"]
    samples = []
    skipped = 0
    for row in data:
        question = row.get(question_field, "").strip()
        answer = row.get(answer_field, "").strip()
        if not question or not answer:
            skipped += 1
            continue
        samples.append(
            QASample(
                id=row.get("id", ""),
                question=question,
                ground_truth=answer,
                domain=row.get("domain", ""),
                difficulty=row.get("difficulty", ""),
                capability=row.get("capability", ""),
                knowledge_text=row.get("knowledge_text", ""),
                metadata={
                    "industry_primary": row.get("industry_primary", ""),
                    "_format": row.get("_format", ""),
                    "grading": row.get("grading", ""),
                    "knowledge_dependency": row.get("knowledge_dependency", ""),
                },
            )
        )
    if skipped > 0:
        print(f"  Skipped {skipped} {LANGUAGE_NAMES[language]} entries (empty question or answer)")
    return samples
