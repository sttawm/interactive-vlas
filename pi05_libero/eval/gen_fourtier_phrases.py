#!/usr/bin/env python3
"""Natural + adversarial tier phrases for the pi0.5/LIBERO four-tier eval.

Faithful port of phrase-rl's tier generator (phrase-rl/scripts/generate_phrases.py
+ prompts/rules_loop/generate.md, used for the SIMPLER bank's natural/adversarial
tiers): text-only, gemini-pro-latest, one call per task producing a NATURAL and
an ADVERSARIAL section. 5 + 5 per task here (SIMPLER used 7 + 2).

  GEMINI_API_KEY=... python3 gen_fourtier_phrases.py   # writes fourtier_phrases.json
"""
import json
import os
import pathlib
import time

from google import genai
from google.genai import types

HERE = pathlib.Path(__file__).parent
N_NAT, N_ADV = 5, 5
MODEL = "gemini-pro-latest"

# prompts/rules_loop/generate.md, verbatim (phrase-rl @ HEAD)
PROMPT = """You are producing test instructions for a robot manipulation task, to measure how
sensitive a policy is to the way a request is worded.

THE TASK'S OWN INSTRUCTION (the wording the policy was trained on):
{instruction}

Write two sets of rewordings that keep the goal EXACTLY the same. The object, the
destination, and the action must be identical in every line — only the wording
changes. A line that changes which object is moved, or where it goes, is useless
to us and worse than no line at all.

## NATURAL ({n_natural} lines)
What a person would actually say to a robot in a kitchen or on a workbench:
fluent, unremarkable, varied in structure. Vary the verb, the article, the word
order and the level of detail across the set — not {n_natural} versions of one
sentence. Some may name the object by colour or material if that is unambiguous.
These should be EASY.

## ADVERSARIAL ({n_adversarial} lines)
Awkward, ornate, or indirect phrasings that a policy is likely to handle badly
while a person would still understand them: heavy subordinate clauses, indirect
reference ("the thing you would drink when thirsty"), unusual register, polite
circumlocution, or front-loaded conditions. Still unambiguous to a human, and
still the SAME goal. These should be HARD.

Reply with exactly this, and nothing else — no commentary, no numbering, no
markdown fences:

NATURAL
<one instruction per line>

ADVERSARIAL
<one instruction per line>"""


def parse(out):
    nat, adv, kind = [], [], None
    for line in out.splitlines():
        up = line.strip().upper()
        if up.startswith("NATURAL"):
            kind = "natural"
            continue
        if up.startswith("ADVERSARIAL"):
            kind = "adversarial"
            continue
        p = line.strip().strip('"').lstrip("-• ").strip()
        if p and kind == "natural":
            nat.append(p)
        elif p and kind == "adversarial":
            adv.append(p)
    return nat, adv


def main():
    tasks = json.load(open(HERE / "fourtier_tasks.json"))
    out_path = HERE / "fourtier_phrases.json"
    phrases = json.load(open(out_path)) if out_path.exists() else {}
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    # one entry per unique canonical string (tasks can share wording across suites)
    for t in tasks:
        canon = t["canonical"]
        if canon in phrases:
            continue
        for attempt in range(6):
            try:
                out = client.models.generate_content(
                    model=MODEL,
                    contents=PROMPT.format(instruction=canon, n_natural=N_NAT, n_adversarial=N_ADV),
                    config=types.GenerateContentConfig(temperature=0.9, max_output_tokens=8000),
                ).text or ""
            except Exception as e:
                print(f"   retry {attempt+1}: {type(e).__name__}")
                time.sleep(min(5 * 2 ** attempt, 60))
                continue
            nat, adv = parse(out)
            if len(nat) >= N_NAT and len(adv) >= N_ADV:
                phrases[canon] = {"natural": nat[:N_NAT], "adversarial": adv[:N_ADV]}
                break
            time.sleep(2)
        else:
            raise SystemExit(f"generation failed for {canon!r}")
        json.dump(phrases, open(out_path, "w"), indent=2)
        print(f"[{len(phrases)}] {canon}")
        for p in phrases[canon]["natural"]:
            print("   nat:", p)
        for p in phrases[canon]["adversarial"]:
            print("   adv:", p)
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
