# IndustryBench

**Source-grounded industrial procurement QA** for LLMs: items are tied to **Chinese national standards (GB/T)** or **structured industrial product records**, with **human-reviewed** English, Russian, and Vietnamese renderings **aligned to the same item ids** as the Chinese source.

- **Paper:** [arXiv:2605.10267](https://arxiv.org/abs/2605.10267)
- **Code & evaluation script:** [GitHub](https://github.com/alibaba-multimodal-industrial-ai/IndustryBench)

## Quick load

```bash
pip install datasets
```

```python
from datasets import load_dataset

ds = load_dataset("alibaba-multimodal-industrial-ai/IndustryBench", split="train")
print(ds[0].keys())
```

## What’s inside

**2,049** items; Chinese source fields plus aligned `question_en` / `answer_en`, `question_ru` / `answer_ru`, `question_vi` / `answer_vi`; `knowledge_text`, `capability`, `difficulty`, `domain`, `industry_primary`, and related columns. Full construction protocol, label definitions, and evaluation rubric are in the **paper** and **appendices**.

## Citation

Use the BibTeX from the [GitHub README](https://github.com/alibaba-multimodal-industrial-ai/IndustryBench#citation) or the paper.

---

**Maintainers:** To avoid the auto-generated metadata table on this dataset page, **do not** put a `---` … `---` YAML block above this file on the Hub. Tags like `task_categories` / `language` in the Hub UI are optional and can be edited under **Dataset card → Edit dataset tags** if you want them gone too.
