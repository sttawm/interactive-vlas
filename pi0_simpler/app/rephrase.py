"""Live instruction rephraser for the CoVer verifier loop.

The CoVer verifier scores π0's action candidates across a *set* of rephrased instructions and
picks the best. For the 7 canonical SimplerEnv tasks the paper ships a fixed rephrase set
(``simpler_rephrased_final_eval_vlm.json``); but in the interactive UI you can type *any* prompt,
which has no pre-generated rephrases. This module fills that gap: given a typed instruction, it
asks Claude for N faithful reworded variants "at boot time" (as the paper describes deployment),
so the verifier has an ensemble to score. Results are memoised by the caller.

Backends:
  - Claude (if `anthropic` is installed and ANTHROPIC_API_KEY is set): N reworded variants that
    preserve the exact objects and goal of a WidowX tabletop instruction.
  - Fallback (no key / any error): returns [] — the verifier then scores the single prompt only.

Kept dependency-light and import-safe (mirrors pi05_libero/app/planner.py) so the runner works
with or without a key.
"""
from __future__ import annotations

import os
import re

REPHRASE_MODEL = os.environ.get("REPHRASE_MODEL", "claude-sonnet-4-6")


def _claude_rephrase(instruction, n):
    """Return up to `n` rephrases from Claude, or raise if unavailable."""
    import anthropic  # raises ImportError if not installed

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    prompt = (
        "You are helping evaluate a robot manipulation policy on a WidowX tabletop. "
        f"Reword the following instruction {n} different ways. Each rewording MUST refer to the "
        "exact same objects and the exact same goal — only vary the wording, phrasing, and word "
        "order (synonyms, politeness, sentence structure). Do NOT change which objects are "
        "involved or what should end up where. Output one rewording per line, no numbering, no "
        "commentary.\n\n"
        f"Instruction: {instruction}"
    )
    msg = client.messages.create(
        model=REPHRASE_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    lines = [re.sub(r"^\s*[-*\d.)]+\s*", "", ln).strip() for ln in text.splitlines()]
    out = [ln for ln in lines if ln]
    return out[:n]


def generate(instruction, n):
    """Return up to `n` rephrases of `instruction`. [] if Claude is unavailable or errors."""
    instruction = (instruction or "").strip()
    if not instruction or n <= 0:
        return []
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _claude_rephrase(instruction, n)
        except Exception:  # ImportError, API error, etc. -> degrade gracefully
            pass
    return []


if __name__ == "__main__":
    import sys

    cmd = " ".join(sys.argv[1:]) or "put the spoon on the towel"
    for i, r in enumerate(generate(cmd, 7), 1):
        print(f"{i}. {r}")
