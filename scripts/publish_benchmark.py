"""Create GitHub-safe benchmark artifacts from a private evaluation report."""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "benchmark-results"

PUBLIC_CASE_FIELDS = (
    "id",
    "category",
    "question",
    "documents_retrieved",
    "source_hit",
    "web_search_used",
    "retry_count",
    "retrieval_confidence",
    "correctness",
    "faithfulness",
    "relevance",
    "behavior",
    "overall_score",
    "critical_failure",
    "latency_seconds",
    "total_tokens",
    "estimated_cost_usd",
)


def sanitize_case(case):
    """Keep reproducibility metrics without publishing answers or corpus text."""
    return {field: case.get(field) for field in PUBLIC_CASE_FIELDS}


def markdown_report(summary, cases):
    metrics = summary["metrics"]
    gates = summary["gates"]
    categories = summary["categories"]
    lines = [
        "# System Benchmark v2 — Full Production Result",
        "",
        f"- Timestamp (UTC): `{summary['timestamp_utc']}`",
        f"- Profile: `{summary['profile']}`",
        f"- Judge: `{summary['judge_model']}`",
        f"- Cases: **{summary['case_count']}**",
        f"- Final gate: **{'PASS' if summary['passed'] else 'FAIL'}**",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value | Gate |",
        "|---|---:|:---:|",
        f"| Overall score | {metrics['overall_score']:.4f} | {'Pass' if gates['overall_score'] else 'Fail'} |",
        f"| Academic QA score | {metrics['academic_qa_score']:.4f} | {'Pass' if gates['academic_qa_score'] else 'Fail'} |",
        f"| Behavior score | {metrics['behavior_score']:.4f} | {'Pass' if gates['behavior_score'] else 'Fail'} |",
        f"| Source hit rate | {metrics['source_hit_rate']:.4f} | {'Pass' if gates['source_hit_rate'] else 'Fail'} |",
        f"| Critical failures | {metrics['critical_failures']} | {'Pass' if gates['critical_failures'] else 'Fail'} |",
        f"| Mean latency | {metrics['mean_latency_seconds']:.4f} s | Observed |",
        f"| P95 latency | {metrics['p95_latency_seconds']:.4f} s | Observed |",
        f"| Web search rate | {metrics['web_search_rate']:.4f} | Observed |",
        f"| Mean retry count | {metrics['mean_retry_count']:.4f} | Observed |",
        f"| Total tokens | {metrics['total_tokens']:,} | Observed |",
        f"| Estimated cost | USD {metrics['estimated_cost_usd']:.6f} | Observed |",
        "",
        "## Category results",
        "",
        "| Category | Cases | Score | Critical failures |",
        "|---|---:|---:|---:|",
    ]
    for category, result in categories.items():
        lines.append(
            f"| {category} | {result['count']} | "
            f"{result['overall_score']:.4f} | {result['critical_failures']} |"
        )

    bands = (
        (">= 0.90", lambda value: value >= 0.90),
        ("0.75–0.90", lambda value: 0.75 <= value < 0.90),
        ("0.50–0.75", lambda value: 0.50 <= value < 0.75),
        ("< 0.50", lambda value: value < 0.50),
    )
    lines += ["", "## Score distribution", ""]
    for label, predicate in bands:
        lines.append(
            f"- {label}: **{sum(predicate(case['overall_score']) for case in cases)}** cases"
        )

    critical = [case for case in cases if case["critical_failure"]]
    lines += [
        "",
        "## Critical failures",
        "",
        "| ID | Category | Score | Source hit | Question |",
        "|---|---|---:|:---:|---|",
    ]
    for case in critical:
        question = str(case["question"]).replace("|", "\\|")
        source = "N/A" if case["source_hit"] is None else str(case["source_hit"])
        lines.append(
            f"| {case['id']} | {case['category']} | {case['overall_score']:.4f} "
            f"| {source} | {question} |"
        )

    lines += [
        "",
        "## Per-case metrics",
        "",
        "| ID | Category | Score | C | F | R | B | Source | Docs | Web | Retry | Latency | Cost (USD) |",
        "|---|---|---:|---:|---:|---:|---:|:---:|---:|:---:|---:|---:|---:|",
    ]
    for case in cases:
        source = "N/A" if case["source_hit"] is None else str(case["source_hit"])
        lines.append(
            f"| {case['id']} | {case['category']} | {case['overall_score']:.4f} "
            f"| {case['correctness']} | {case['faithfulness']} | {case['relevance']} "
            f"| {case['behavior']} | {source} | {case['documents_retrieved']} "
            f"| {case['web_search_used']} | {case['retry_count']} "
            f"| {case['latency_seconds']:.3f} | {case['estimated_cost_usd']:.6f} |"
        )

    lines += [
        "",
        "## Publication policy",
        "",
        "This public artifact intentionally excludes generated answers, judge reasoning,",
        "ground truth text, and retrieved context. The private JSON report remains under",
        "`reports/system_benchmark/` and is excluded from Git.",
        "",
    ]
    return "\n".join(lines)


def publish(input_path, output_dir):
    report = json.loads(Path(input_path).read_text(encoding="utf-8"))
    summary = report["summary"]
    cases = [sanitize_case(case) for case in report["cases"]]
    stamp = summary["timestamp_utc"][:10]
    stem = f"{stamp}-{summary['profile']}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"

    markdown_path.write_text(markdown_report(summary, cases), encoding="utf-8")
    json_path.write_text(
        json.dumps({"summary": summary, "cases": cases}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PUBLIC_CASE_FIELDS)
        writer.writeheader()
        writer.writerows(cases)
    return markdown_path, json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for path in publish(args.report, args.output_dir):
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
