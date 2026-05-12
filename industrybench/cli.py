import argparse
import os
import sys

from .data import extract_language_samples, load_dataset
from .languages import LANGUAGE_NAMES, LANGUAGE_PROMPTS
from .runner import run_single_model


def main():
    parser = argparse.ArgumentParser(description="IndustryBench Evaluation Script")
    parser.add_argument("--data-path", type=str, default="huggingface_dataset.csv", help="Path to dataset CSV file")
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        choices=["zh", "en", "ru", "vi", "all"],
        help="Language to evaluate (default: zh)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        required=True,
        help="OpenAI-compatible API base URL (e.g. https://api.openai.com/v1)",
    )
    parser.add_argument("--api-key", type=str, default=None, help="API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, required=True, help="Model name to evaluate")
    parser.add_argument("--judge-model", type=str, default=None, help="Judge model for scoring (default: same as --model)")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrency (default: 5)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--no-safety-review", action="store_true", help="Disable safety review")
    parser.add_argument("--prompt-template", type=str, default=None, help="Custom prompt with ${question} placeholder")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature (default: 0.0)")
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        alt = os.path.join(repo_root, data_path)
        if os.path.exists(alt):
            data_path = alt
        else:
            print(f"Error: dataset file not found: {data_path}")
            sys.exit(1)

    print(f"[1/4] Loading dataset: {data_path}")
    data = load_dataset(data_path)
    print(f"  Total records: {len(data)}")

    languages = ["zh", "en", "ru", "vi"] if args.language == "all" else [args.language]

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: API key required. Set --api-key or OPENAI_API_KEY env var.")
        sys.exit(1)

    judge_model = args.judge_model or args.model

    for lang in languages:
        print(f"\n{'='*80}")
        print(f"  Evaluating: {LANGUAGE_NAMES[lang]} ({lang}) | Model: {args.model}")
        print(f"{'='*80}\n")

        samples = extract_language_samples(data, lang)
        if not samples:
            print(f"  No {LANGUAGE_NAMES[lang]} data found, skipping")
            continue

        print(f"  Samples: {len(samples)}")

        prompt_template = args.prompt_template or LANGUAGE_PROMPTS.get(lang, LANGUAGE_PROMPTS["zh"])

        run_single_model(
            model_name=args.model,
            samples=samples,
            api_key=api_key,
            api_base=args.api_base,
            judge_model=judge_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            prompt_template=prompt_template,
            enable_safety_review=not args.no_safety_review,
            concurrency=args.concurrency,
            checkpoint_file=os.path.join(args.output_dir, f"checkpoint_{args.model.replace('/', '_')}_{lang}.json"),
            output_dir=args.output_dir,
        )
        print(f"\n  {LANGUAGE_NAMES[lang]} evaluation complete")

    print(f"\n{'='*80}")
    print("  All evaluations complete")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
