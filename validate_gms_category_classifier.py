from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from gmm_predict import get_available_categories
from hybrid_category_classifier import GMSCategoryClassifier


DEFAULT_SAMPLES_PATH = BASE_DIR / "analysis" / "gms_category_eval_samples.json"


def load_samples(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(samples: list[dict]) -> dict:
    classifier = GMSCategoryClassifier(get_available_categories())
    if not classifier.ready:
        raise RuntimeError(
            "GMS classifier is not ready. Check CATEGORY_LLM_ENABLED and GMS_KEY in .env."
        )

    results: list[dict] = []
    started = time.perf_counter()

    for index, sample in enumerate(samples, start=1):
        tick = time.perf_counter()
        decision = classifier.classify(
            merchant_name=str(sample["merchant_name"]),
            transaction_detail=str(sample.get("transaction_detail", "")),
            payment_method=str(sample.get("payment_method", "카드")),
            amount=int(sample.get("amount", 0)),
        )
        latency_ms = round((time.perf_counter() - tick) * 1000, 1)
        predicted = decision.category if decision else None
        confidence = decision.confidence if decision else None
        reason = decision.reason if decision else "No decision returned"
        expected = str(sample["expected_category"])
        matched = predicted == expected

        results.append(
            {
                "index": index,
                "merchant_name": sample["merchant_name"],
                "transaction_detail": sample.get("transaction_detail", ""),
                "expected_category": expected,
                "predicted_category": predicted,
                "matched": matched,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "reason": reason,
            }
        )

    total = len(results)
    correct = sum(1 for item in results if item["matched"])
    mismatches = [item for item in results if not item["matched"]]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    return {
        "summary": {
            "model": classifier.model,
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "elapsed_ms": elapsed_ms,
            "avg_latency_ms": round(sum(item["latency_ms"] for item in results) / total, 1)
            if total
            else 0.0,
        },
        "results": results,
        "mismatches": mismatches,
    }


def print_report(report: dict, only_misses: bool) -> None:
    summary = report["summary"]
    print("[summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()

    rows = report["mismatches"] if only_misses else report["results"]
    label = "mismatches" if only_misses else "results"
    print(f"[{label}]")
    for item in rows:
        status = "OK" if item["matched"] else "MISS"
        print(
            f"{status} | {item['merchant_name']} | expected={item['expected_category']} | "
            f"predicted={item['predicted_category']} | confidence={item['confidence']} | "
            f"latency_ms={item['latency_ms']}"
        )
        print(f"reason: {item['reason']}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GMS category classifier accuracy.")
    parser.add_argument(
        "--samples",
        type=Path,
        default=DEFAULT_SAMPLES_PATH,
        help="Path to a JSON file containing labeled merchant samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the full evaluation report as JSON.",
    )
    parser.add_argument(
        "--only-misses",
        action="store_true",
        help="Print only mismatched cases.",
    )
    args = parser.parse_args()

    samples = load_samples(args.samples)
    report = evaluate(samples)
    print_report(report, only_misses=args.only_misses)

    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
