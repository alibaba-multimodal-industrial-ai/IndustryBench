<div align="center">

# IndustryBench

**A Multi-lingual Benchmark for Probing the Industrial Knowledge Boundaries of LLMs**

[📄 Paper](https://arxiv.org/abs/2605.10267) · [🤗 Dataset](https://huggingface.co/datasets/danielbai1703/IndustryBench)

</div>

## Overview

IndustryBench is a benchmark for evaluating the industrial domain knowledge of large language models. It consists of **2,049** expert-curated QA pairs with human-reviewed translations across **four languages** (Chinese, English, Russian, Vietnamese).

| | |
|---|---|
| Questions | 2,049 |
| Languages | zh, en, ru, vi |
| Industry categories | 10 |
| Capability dimensions | 7 |
| Difficulty | easy / medium / hard |

## Installation

```bash
pip install huggingface_hub
```

## Usage

### Load the dataset

```python
from datasets import load_dataset

dataset = load_dataset("danielbai1703/IndustryBench")
df = dataset["train"].to_pandas()
```

### Run evaluation

The included `evaluate.py` script runs multi-lingual QA evaluation with LLM-as-Judge scoring (0–3) and safety review. It accepts any OpenAI-compatible API endpoint.

```bash
# Evaluate on Chinese questions
python evaluate.py \
  --language zh \
  --api-base https://api.openai.com/v1 \
  --api-key $OPENAI_API_KEY \
  --model qwen3-max

# Evaluate all four languages
python evaluate.py \
  --language all \
  --api-base https://api.openai.com/v1 \
  --api-key $OPENAI_API_KEY \
  --model qwen3-max
```

Run `python evaluate.py --help` for all available options.

### Evaluation protocol

Each model response is scored by an LLM judge on a 0–3 scale:

| Score | Criterion |
|-------|-----------|
| 3 | Consistent with ground truth, matching reasoning and logic |
| 2 | Largely correct but differs in reasoning |
| 1 | Partially correct or plausible reasoning with wrong conclusion |
| 0 | Entirely incorrect |

A safety review module additionally checks for industrial safety violations (e.g., non-compliant equipment recommendations, missing safety procedures). Violations result in a final score of 0.

## Dataset Structure

Each row contains the following fields:

| Field | Description |
|-------|-------------|
| `id` | Unique ID (1–2049) |
| `question` / `answer` | Chinese QA pair |
| `question_en` / `answer_en` | English QA pair |
| `question_ru` / `answer_ru` | Russian QA pair |
| `question_vi` / `answer_vi` | Vietnamese QA pair |
| `difficulty` | easy / medium / hard |
| `_format` | Question format |
| `industry_primary` | Industry category |
| `capability` | Capability dimension |
| `knowledge_text` | Reference text from national/industry standard or product manual |
| `domain` | Broad domain label |

### Capability dimensions

Each item receives one primary capability label:

- **Selection & Substitution** (31.7%)
- **Standards & Terminology** (29.8%)
- **Process Principles** (25.7%)
- **Safety & Compliance** (5.7%)
- **Quality & Metrology** (4.5%)
- **Fault Diagnosis** (1.5%)
- **Engineering Calculation** (1.1%)

### Industry categories

Ten major industrial product verticals:

- Machinery & Hardware (23.3%)
- Chemical & Coatings (19.8%)
- Electronics & Sensors (16.2%)
- Electrical & Power (11.7%)
- Cross-Industry (9.3%)
- Metallurgy & Mining (5.9%)
- Energy & Storage (4.1%)
- Security & Fire Safety (3.7%)
- Packaging & Printing (3.7%)
- Textile & Leather (2.4%)

The distributions reflect the natural frequency of the verified source pool rather than a deliberately balanced design.

## Citation

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
