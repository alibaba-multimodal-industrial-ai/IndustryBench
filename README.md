# IndustryBench

Multi-lingual industrial QA benchmark (2,049 items, zh / en / ru / vi).

[Paper](https://arxiv.org/abs/2605.10267) · [Dataset](https://huggingface.co/datasets/alibaba-multimodal-industrial-ai/IndustryBench)

## Dataset

```bash
pip install datasets
```

```python
from datasets import load_dataset
ds = load_dataset("alibaba-multimodal-industrial-ai/IndustryBench", split="train")
ds.to_csv("industrybench.csv")  # for local evaluation
```

## Evaluation

`--api-base` is the **OpenAI-compatible API root** (must end with `/v1`). The script calls `POST {api-base}/chat/completions`. It is **not** the model name; the model is `--model` (and optionally `--judge-model`).

**DashScope / Qwen:** use the compatible-mode base URL from the [DashScope OpenAI compatibility](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope) documentation, for example `https://dashscope.aliyuncs.com/compatible-mode/v1`, plus your DashScope `sk-` key.

```bash
pip install -r requirements.txt
export DASHSCOPE_API_KEY="sk-..."   # or use --api-key

python evaluate.py --data-path industrybench.csv --language zh \
  --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --model qwen3-max

# Optional: another model answers, qwen3-max judges (and safety review)
python evaluate.py --data-path industrybench.csv --language zh \
  --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --model <your_answer_model> --judge-model qwen3-max
```

**OpenAI:**

```bash
python evaluate.py --data-path industrybench.csv --language zh \
  --api-base https://api.openai.com/v1 --api-key "$OPENAI_API_KEY" --model gpt-4o
```

`python evaluate.py --help` for all options.

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
