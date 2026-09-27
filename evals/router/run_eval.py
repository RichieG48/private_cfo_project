"""
Router evaluation: measures how well SovereignRouter keeps sensitive queries local.

Usage:
    PYTHONPATH=. python evals/router/run_eval.py [--limit N] [--out PATH]

Headline metric is the LEAK RATE: the share of sensitive queries routed to the cloud.
"""
import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.config import settings
from src.core.router import SovereignRouter

EVAL_DIR = Path(__file__).resolve().parent
QUERIES_PATH = EVAL_DIR / "queries.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


def load_queries(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(results: list[dict]) -> dict:
    sensitive = [r for r in results if r["label"] == "sensitive"]
    generic = [r for r in results if r["label"] == "generic"]
    leaks = [r for r in sensitive if r["predicted"] == "generic"]
    over_cautious = [r for r in generic if r["predicted"] == "sensitive"]

    by_category = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_category[r["category"]]["total"] += 1
        by_category[r["category"]]["correct"] += r["correct"]

    latencies = sorted(r["latency_s"] for r in results)
    return {
        "total": len(results),
        "accuracy": rate(sum(r["correct"] for r in results), len(results)),
        "leak_rate": rate(len(leaks), len(sensitive)),
        "leaks": len(leaks),
        "sensitive_total": len(sensitive),
        "over_caution_rate": rate(len(over_cautious), len(generic)),
        "over_cautious": len(over_cautious),
        "generic_total": len(generic),
        "by_category": {
            cat: {**v, "accuracy": rate(v["correct"], v["total"])}
            for cat, v in sorted(by_category.items())
        },
        "latency_median_s": latencies[len(latencies) // 2] if latencies else 0.0,
    }


def print_report(summary: dict, results: list[dict]) -> None:
    print(f"\nRouter eval: {summary['total']} queries, model={settings.LOCAL_MODEL_NAME}\n")
    print(f"  Leak rate (sensitive -> cloud):   {summary['leak_rate']:6.1%}  "
          f"({summary['leaks']}/{summary['sensitive_total']})")
    print(f"  Over-caution (generic -> local):  {summary['over_caution_rate']:6.1%}  "
          f"({summary['over_cautious']}/{summary['generic_total']})")
    print(f"  Accuracy:                         {summary['accuracy']:6.1%}")
    print(f"  Median latency:                   {summary['latency_median_s']:.2f}s\n")

    print("  By category:")
    for cat, v in summary["by_category"].items():
        print(f"    {cat:18} {v['accuracy']:6.1%}  ({v['correct']}/{v['total']})")

    misroutes = [r for r in results if not r["correct"]]
    if misroutes:
        print("\n  Misroutes:")
        for r in misroutes:
            marker = "LEAK" if r["label"] == "sensitive" else "over"
            print(f"    [{marker}] {r['id']}: {r['query']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="Only run the first N queries.")
    parser.add_argument("--out", type=Path, help="Where to write the JSON results.")
    args = parser.parse_args()

    # Silence per-query router logs; the report below covers them.
    logging.getLogger("src.core.router").setLevel(logging.ERROR)

    queries = load_queries(QUERIES_PATH)[: args.limit]
    router = SovereignRouter()

    results = []
    for i, q in enumerate(queries, 1):
        start = time.perf_counter()
        predicted = router.route_query(q["query"])
        latency = time.perf_counter() - start
        results.append({**q, "predicted": predicted, "correct": predicted == q["label"], "latency_s": round(latency, 3)})
        print(f"\r  {i}/{len(queries)}", end="", flush=True)

    summary = summarize(results)
    print_report(summary, results)

    out = args.out or RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": settings.LOCAL_MODEL_NAME, "summary": summary, "results": results}, indent=2))
    print(f"  Results written to {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")


if __name__ == "__main__":
    main()
