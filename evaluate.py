#!/usr/bin/env python3
"""
IndustryBench evaluation CLI (OpenAI-compatible API).

Implementation lives in the ``industrybench`` package; this file is the stable entrypoint:

    python evaluate.py --language zh --api-base https://api.openai.com/v1 --api-key $OPENAI_API_KEY --model gpt-4o

Run ``python evaluate.py --help`` for options.
"""

from industrybench.cli import main

if __name__ == "__main__":
    main()
