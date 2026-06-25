"""Optional planner wrapper: decompose a high-level command into ordered subgoals.

This supports the experiment of comparing pi0.5 with **one long prompt** vs. pi0.5
fed **staged subgoals** by an external planner. The interactive runner can take the
subgoal list and feed pi0.5 one subgoal at a time.

Two backends:
  - Claude (if `anthropic` is installed and ANTHROPIC_API_KEY is set): turns a goal
    like "set the table" into concrete LIBERO-style subgoals, grounded in the objects
    actually present in the scene.
  - Heuristic fallback (no key needed): splits on "then"/"and then"/", then"/";" and
    otherwise returns the command unchanged as a single subgoal.

Kept dependency-light and import-safe so the runner works with or without a key.
"""
from __future__ import annotations

import os
import re

PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "claude-sonnet-4-6")

_SPLIT_RE = re.compile(r"\s*(?:;|,?\s*then\b|\band then\b)\s*", flags=re.IGNORECASE)


def _heuristic_plan(command):
    parts = [p.strip() for p in _SPLIT_RE.split(command) if p.strip()]
    return parts or [command.strip()]


def _claude_plan(command, scene_objects, task_language):
    """Return subgoals from Claude, or raise if unavailable."""
    import anthropic  # raises ImportError if not installed

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    ctx = []
    if task_language:
        ctx.append(f"The scene's canonical task is: {task_language!r}.")
    if scene_objects:
        ctx.append("Objects visible in the scene: " + ", ".join(scene_objects) + ".")
    context = (" ".join(ctx)) or "Assume a typical LIBERO tabletop manipulation scene."

    prompt = (
        "You are a planner for a single-arm robot in the LIBERO simulator. "
        "Decompose the user's command into the smallest sequence of concrete, single-action "
        "subgoals a low-level policy can execute (e.g. 'open the top drawer', "
        "'pick up the bowl', 'put the bowl in the drawer', 'close the drawer'). "
        "Only use objects that exist in the scene. Output one subgoal per line, no numbering, "
        "no commentary.\n\n"
        f"Scene context: {context}\n"
        f"Command: {command}"
    )
    msg = client.messages.create(
        model=PLANNER_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
    subgoals = [ln.strip(" -*\t") for ln in text.splitlines() if ln.strip()]
    return subgoals or [command.strip()]


def plan(command, scene_objects=None, task_language=None):
    """Decompose `command` into an ordered list of subgoal strings.

    Falls back to the heuristic splitter if Claude is unavailable or errors.
    """
    command = (command or "").strip()
    if not command:
        return []
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _claude_plan(command, scene_objects, task_language)
        except Exception:  # ImportError, API error, etc. -> degrade gracefully
            pass
    return _heuristic_plan(command)


if __name__ == "__main__":
    import sys

    cmd = " ".join(sys.argv[1:]) or "open the drawer then put the bowl in the drawer then close it"
    for i, sg in enumerate(plan(cmd), 1):
        print(f"{i}. {sg}")
