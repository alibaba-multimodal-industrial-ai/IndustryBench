# IndustryBench

Multi-lingual industrial QA benchmark (2,049 items, zh / en / ru / vi).

[Paper](https://arxiv.org/abs/2605.10267) · [Dataset](https://huggingface.co/datasets/danielbai1703/IndustryBench)

## Dataset

```bash
pip install datasets
```

```python
from datasets import load_dataset
ds = load_dataset("danielbai1703/IndustryBench", split="train")
ds.to_csv("industrybench.csv")  # for local evaluation
```

## Evaluation

```bash
pip install -r requirements.txt
python evaluate.py --data-path industrybench.csv --language zh \
  --api-base https://api.openai.com/v1 --api-key "$OPENAI_API_KEY" --model <model_id>
```

`python evaluate.py --help` for options.

See `LICENSE` (MIT).

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
