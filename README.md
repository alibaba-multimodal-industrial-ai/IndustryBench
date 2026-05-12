---
language:
- zh
- en
- ru
- vi
license: mit
task_categories:
- question-answering
- text-generation
pretty_name: IndustryBench
---

# IndustryBench: Probing the Industrial Knowledge Boundaries of LLMs

**IndustryBench** is a multi-lingual benchmark for evaluating the industrial domain knowledge of large language models. It comprises **2,049** expert-curated QA pairs spanning **12** industrial sectors, with human-reviewed translations in **Chinese, English, Russian, and Vietnamese**.

## Overview

| Dimension | Details |
|-----------|---------|
| Total questions | 2,049 |
| Languages | Chinese (zh), English (en), Russian (ru), Vietnamese (vi) |
| Difficulty | Easy / Medium / Hard |
| Industry sectors | 10 (Machinery & Hardware, Chemical & Coatings, Electronics & Sensors, Electrical & Power, Cross-Industry, Metallurgy & Mining, Energy & Storage, Security & Fire Safety, Packaging & Printing, Textile & Leather) |
| Capability tags | 7 (Selection & Substitution 31.7%, Standards & Terminology 29.8%, Process Principles 25.7%, Safety & Compliance 5.7%, Quality & Metrology 4.5%, Fault Diagnosis 1.5%, Engineering Calculation 1.1%) |
| Knowledge grounding | Each question is linked to a national/industry standard, product manual, or authoritative reference |

## Dataset Structure

Each row in the dataset contains the following fields:

| Field | Description |
|-------|-------------|
| `id` | Unique question ID (1–2049) |
| `question` | Chinese question |
| `answer` | Chinese ground-truth answer |
| `question_en` | English question |
| `answer_en` | English ground-truth answer |
| `question_ru` | Russian question |
| `answer_ru` | Russian ground-truth answer |
| `question_vi` | Vietnamese question |
| `answer_vi` | Vietnamese ground-truth answer |
| `difficulty` | Difficulty level: `easy`, `medium`, `hard` |
| `_format` | Question format |
| `industry_primary` | Primary industry classification |
| `capability` | Capability tag |
| `knowledge_text` | Reference text from the relevant standard, manual, or authoritative source |
| `domain` | Broad domain label |

## Industry Sectors

The dataset covers 10 major industrial product verticals: Machinery & Hardware (23.3%), Chemical & Coatings (19.8%), Electronics & Sensors (16.2%), Electrical & Power (11.7%), Cross-Industry (9.3%), Metallurgy & Mining (5.9%), Energy & Storage (4.1%), Security & Fire Safety (3.7%), Packaging & Printing (3.7%), and Textile & Leather (2.4%). The distribution reflects the verified source pool rather than a deliberately balanced design.

## Capability Dimensions

Each item is annotated with one primary capability label across 7 dimensions:

- **Selection & Substitution** (31.7%) — Equipment and material selection, cross-product substitution
- **Standards & Terminology** (29.8%) — Knowledge of national/industry standards and technical terminology
- **Process Principles** (25.7%) — Understanding of industrial processes and their parameters
- **Safety & Compliance** (5.7%) — Safety regulations, risk control, compliance requirements
- **Quality & Metrology** (4.5%) — Measurement, inspection, and quality assurance
- **Fault Diagnosis** (1.5%) — Equipment fault identification and resolution
- **Engineering Calculation** (1.1%) — Quantitative engineering computations

The distribution follows the natural frequency of the verified source pool. Selection, standards, and process questions dominate because they are more prevalent in the source material. Fault Diagnosis (31 questions) and Engineering Calculation (22 questions) have limited support, so per-dimension findings on these two labels should be interpreted as diagnostic signals rather than precise rankings.

## Evaluation Protocol

IndustryBench uses an **LLM-as-Judge** evaluation pipeline with a 0–3 scoring rubric:

| Score | Criterion |
|-------|-----------|
| 3 | Model answer is essentially consistent with the ground truth, with matching reasoning and logic |
| 2 | Model answer is largely correct but differs in reasoning or justification |
| 1 | Model answer is partially correct or has plausible reasoning but incorrect conclusion |
| 0 | Model answer is entirely incorrect |

Additionally, a **safety review** module checks whether the model response contains industrial safety violations (e.g., recommending non-compliant equipment, omitting critical safety steps, violating mandatory standard clauses). Responses flagged as safety violations receive a final score of 0 regardless of the judge score.

## Quick Start

### Load the dataset

```python
from datasets import load_dataset

dataset = load_dataset("danielbai1703/IndustryBench")
df = dataset["train"].to_pandas()
```

### Run evaluation

```bash
# Chinese evaluation
python evaluate.py --language zh --api-base https://api.openai.com/v1 --api-key $OPENAI_API_KEY --model gpt-4o

# English evaluation
python evaluate.py --language en --api-base https://api.openai.com/v1 --api-key $OPENAI_API_KEY --model gpt-4o

# All four languages
python evaluate.py --language all --api-base https://api.openai.com/v1 --api-key $OPENAI_API_KEY --model gpt-4o
```

See `evaluate.py --help` for all options.

## Evaluation Script

The included `evaluate.py` script provides:

- Multi-language support (zh / en / ru / vi) with language-specific prompts
- LLM-as-Judge scoring (0–3) with configurable judge model
- Safety & compliance review using knowledge_text as authoritative reference
- Checkpoint-based resume for large-scale evaluation
- Per-model CSV output with detailed statistics

## Data Quality

All 2,049 questions are expert-curated with human-reviewed translations across all four languages. The `knowledge_text` field links each question to its authoritative reference source (national standard, industry standard, or product manual).

## Citation

If you use this dataset in your work, please cite:

```bibtex
@misc{bai2026industrybenchprobingindustrialknowledge,
  title={IndustryBench: Probing the Industrial Knowledge Boundaries of LLMs},
  author={Songlin Bai and Xintong Wang and Linlin Yu and Bin Chen and Zhiang Xu and Yuyang Sheng and Changtong Zan and Xiaofeng Zhu and Yizhe Zhang and Jiru Li and Mingze Guo and Ling Zou and Yalong Li and Chengfu Huo and Liang Ding},
  year={2026},
  eprint={2605.10267},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2605.10267},
}
```

## License

MIT
