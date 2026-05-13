# IndustryBench: Probing the Industrial Knowledge Boundaries of LLMs

[📝Paper](https://arxiv.org/abs/2605.10267) | [🤗HuggingFace Data](https://huggingface.co/datasets/alibaba-multimodal-industrial-ai/IndustryBench)  

**Source-grounded industrial procurement QA** for LLMs: each item is tied to a **Chinese national standard (GB/T)** or a **structured industrial product record**, with **human-reviewed** English, Russian, and Vietnamese renderings **aligned to the same item ids** as the Chinese source.

| | |
|:---|:---|
| **Items** | 2,049 |
| **Languages** | Chinese (source) + EN / RU / VI (aligned) |
| **Labels** | 7 capability dimensions · 10 industry categories · panel-derived difficulty (easy / medium / hard) |
| **Sources** | GB/T excerpts + industrial product records (see paper §3) |
| **Paper** | [arXiv:2605.10267](https://arxiv.org/abs/2605.10267) |
| **Dataset** | [`alibaba-multimodal-industrial-ai/IndustryBench`](https://huggingface.co/datasets/alibaba-multimodal-industrial-ai/IndustryBench) on Hugging Face |

**Evaluation idea (paper §4):** models answer **closed-book** from the question only; a **calibrated LLM judge** scores **raw** correctness on **0–3**; a **separate safety-violation (SV)** check uses the **source excerpt** (`knowledge_text`). SV hits can zero the effective score—see paper for the full protocol and human calibration (**κ_w ≈ 0.798** on the judge sample).

---

## Who this repo is for

| You want… | Do this |
|:---|:---|
| **Only the data** | Use Hugging Face below—no clone required. |
| **The same scoring pipeline as the paper** | Clone this repo, export a CSV, run `evaluate.py` (below). |

---

## 1. Load the dataset (most users)

```bash
pip install datasets
```

```python
from datasets import load_dataset

ds = load_dataset("alibaba-multimodal-industrial-ai/IndustryBench", split="train")
# e.g. inspect
print(ds[0].keys())
```

The Hugging Face UI may show a small **metadata** table (language, license, task tags, etc.) if the dataset `README.md` on the Hub starts with a **`---` YAML block**. That block is optional; a YAML-free template lives in this repo at [`huggingface/README.md`](huggingface/README.md) for you to paste on the Hub if you want that table gone.

Typical columns include `question` / `answer` (Chinese), `question_en` / `answer_en`, `question_ru` / `answer_ru`, `question_vi` / `answer_vi`, `knowledge_text`, `capability`, `difficulty`, `domain`, `industry_primary`, etc. Full schema is documented in the **paper appendix** and on the **HF dataset card** body (markdown below any YAML).

---

## 2. Reproduce the released evaluation script

**Prereqs:** Python 3.10+, `pip install -r requirements.txt`, and an **OpenAI-compatible** HTTP API (any host that exposes `POST …/v1/chat/completions`).

**Steps**

1. Export the HF split to CSV (path can be anything; used as `--data-path`):

   ```python
   from datasets import load_dataset
   load_dataset("alibaba-multimodal-industrial-ai/IndustryBench", split="train").to_csv("industrybench.csv")
   ```

2. Set an API key (`--api-key` **or** env `OPENAI_API_KEY` **or** `DASHSCOPE_API_KEY`).

3. Run (example: DashScope-compatible base + Qwen):

   ```bash
   python evaluate.py \
     --data-path industrybench.csv \
     --language zh \
     --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
     --model qwen3-max
   ```

   - **`--api-base`** — Root URL that ends with **`/v1`**. The script appends **`/chat/completions`** itself. It is **not** the model name.
   - **`--model`** — Model that **answers** the questions.
   - **`--judge-model`** — Optional; defaults to `--model`. Set to your judge (e.g. `qwen3-max`) if the answer model differs.

4. Results and checkpoints go under `results/` by default. See `python evaluate.py --help`.

---

## 3. What’s in this repository

| Path | Role |
|:---|:---|
| `evaluate.py` | End-to-end multilingual runner: generation → LLM judge (0–3) → optional safety review → CSV. |
| `requirements.txt` | Minimal Python deps for `evaluate.py`. |
| `huggingface/README.md` | Suggested **Hub dataset card** (no YAML frontmatter); paste on HF to drop the auto metadata table. |
| `LICENSE` | MIT |

Large raw CSVs are **not** stored in git; the canonical release is **Hugging Face**.

---

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

Field-level schema and construction details: **Hugging Face dataset card** + **paper appendix**.
