"""Single-command, reproducible demo of the full pipeline.

Run with:
    python scripts/run_demo.py

This regenerates the synthetic corpus, runs Drain at the conservative
operating threshold, rebuilds the candidate merge-pair dataset, replays the
cached (real, Claude-produced) LLM judge over it, applies the LLM-calibrated
merge policy, and runs drift detection, printing a human-readable report.
It does not require an ANTHROPIC_API_KEY or network access: the judge step
replays data/llm_judgments_cache.json.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("=" * 72)
    print("Step 1/2: rebuilding the candidate merge-pair dataset from a fresh")
    print("          synthetic corpus and the Drain parser")
    print("=" * 72)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_dataset.py")], check=True)

    print()
    print("=" * 72)
    print("Step 2/2: evaluating the LLM-calibrated merge policy and drift")
    print("          detector, writing data/evaluation_report.json")
    print("=" * 72)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py")], check=True)

    print()
    print("Done. See data/evaluation_report.json for full per-pair detail,")
    print("or README.md for the narrative summary of these same numbers.")


if __name__ == "__main__":
    main()
