import re
import sys
import time
import types
from pathlib import Path


# Mock IPython before any neurogolf_utils import
if "IPython" not in sys.modules:
    _mock_ipython = types.ModuleType("IPython")
    _mock_display = types.SimpleNamespace(
        display=lambda *a, **k: None,
        FileLink=lambda *a, **k: None,
    )
    _mock_ipython.display = _mock_display
    sys.modules["IPython"] = _mock_ipython
    sys.modules["IPython.display"] = _mock_display

try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass

from crewai import Crew, Process

import neurogolf_utils
from agents import build_llm, create_agents
from tasks import create_tasks
from tools import _load_task


OUTPUT_DIR = Path("outputs")
NUM_TASKS = 1
MAX_ATTEMPTS = 1
TARGET_SCORE = 20.0
_MAX_RATE_LIMIT_RETRIES = 1
_RATE_LIMIT_BASE_SLEEP = 35  # seconds — slightly above the observed 29s retry_after



def _is_rate_limit(exc: Exception) -> int | None:
    """
    Return the number of seconds to sleep if `exc` is a 429 rate-limit error,
    or None if it's a different kind of error.
    Tries to honour the provider's retry_after_seconds hint.
    """
    msg = str(exc)
    if "429" not in msg and "rate" not in msg.lower():
        return None
    m = re.search(r"retry_after_seconds[_a-z]*['\"]?\s*[:\s]+([0-9]+\.?[0-9]*)", msg)
    if m:
        return max(int(float(m.group(1))) + 5, 10)
    return _RATE_LIMIT_BASE_SLEEP


def _kickoff_with_backoff(crew) -> object:
    """
    Run crew.kickoff(), transparently retrying on 429 rate-limit responses
    with exponential backoff.  Does NOT consume an attempt slot.
    """
    for retry in range(_MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return crew.kickoff()
        except Exception as exc:
            sleep_secs = _is_rate_limit(exc)
            if sleep_secs is None or retry >= _MAX_RATE_LIMIT_RETRIES:
                raise
            # Double the base sleep each retry so we back off progressively
            wait = sleep_secs * (2 ** retry)
            print(
                f"  [RATE LIMIT 429] Sleeping {wait}s before transparent retry "
                f"{retry + 1}/{_MAX_RATE_LIMIT_RETRIES} (attempt slot not consumed)..."
            )
            time.sleep(wait)
    raise RuntimeError("Exhausted rate-limit retries")


def parse_score_from_audit(text: str) -> float:
    """Extract the float score from the Auditor's output text."""
    match = re.search(r"score[:\s]+([0-9]+\.?[0-9]*)", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 0.0


def parse_pass_from_audit(text: str) -> bool:
    """Check if the Auditor's output indicates all examples passed."""
    first_line = text.strip().splitlines()[0].upper() if text.strip() else ""
    if "PASS" in first_line:
        return True
    lower = text.lower()
    return (
        "all examples pass" in lower
        or "network is ready" in lower
        or ("pass" in lower and "fail" not in lower[:200])
    )


def solve_task(task_num: int, agents: dict) -> tuple:
    """
    Run up to MAX_ATTEMPTS to solve a single ARC-AGI task.
    Returns (best_score, best_model_path).
    """
    best_score = 0.0
    best_model_path = None
    feedback = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        model_path = OUTPUT_DIR / f"task_{task_num:03d}_attempt_{attempt:02d}.onnx"
        print(f"\n  [Task {task_num} | Attempt {attempt}/{MAX_ATTEMPTS}] "
              f"Target: {model_path.name}")

        tasks = create_tasks(
            agents=agents,
            task_num=task_num,
            attempt=attempt,
            model_output_path=str(model_path),
            feedback=feedback,
            best_score=best_score,
        )

        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        try:
            result = _kickoff_with_backoff(crew)

            # Extract Auditor's output (last task)
            audit_text = ""
            if result.tasks_output:
                audit_text = result.tasks_output[-1].raw or ""

            score = parse_score_from_audit(audit_text)
            passed = parse_pass_from_audit(audit_text)

            print(f"  Passed={passed} | Score={score:.3f}")

            if passed and score > best_score:
                best_score = score
                best_model_path = str(model_path)

            if passed and score >= TARGET_SCORE:
                print(f"  Target reached ({score:.3f} >= {TARGET_SCORE}). Moving on.")
                break

            feedback = audit_text

        except Exception as e:
            print(f"  [ERROR] Crew failed on attempt {attempt}: {e}")
            feedback = f"Previous attempt {attempt} failed with system error: {e}"

    return best_score, best_model_path


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    llm_analyst = build_llm("analyst")
    llm_architect = build_llm("architect")
    llm_auditor = build_llm("auditor")

    agents = create_agents(llm_analyst, llm_architect, llm_auditor)

    results: dict[int, tuple] = {}

    for task_num in range(1, NUM_TASKS + 1):
        print(f"\n{'=' * 60}")
        print(f"Solving Task {task_num}/{NUM_TASKS}")
        print(f"{'=' * 60}")

        try:
            _load_task(task_num)
        except Exception as e:
            print(f"  [SKIP] Task {task_num}: JSON not found — {e}")
            results[task_num] = (0.0, None)
            continue

        score, model_path = solve_task(task_num, agents)
        results[task_num] = (score, model_path)
        print(f"Task {task_num} complete — best score: {score:.3f} | model: {model_path}")

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    total_score = sum(s for s, _ in results.values())
    solved = sum(1 for s, _ in results.values() if s >= TARGET_SCORE)
    print(f"Total score: {total_score:.3f}")
    print(f"Tasks reaching target (>= {TARGET_SCORE}): {solved}/{NUM_TASKS}")
    for tn, (sc, mp) in sorted(results.items()):
        status = "OK" if sc >= TARGET_SCORE else "  "
        print(f"  [{status}] Task {tn:03d}: {sc:.3f}  {mp or 'no model'}")


if __name__ == "__main__":
    main()
