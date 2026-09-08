"""Cost-aware system-level benchmark for the UIT RAG chatbot."""

import argparse
import csv
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "data" / "benchmark" / "benchmark_config.json"
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports" / "system_benchmark"


def arguments():
    parser = argparse.ArgumentParser(description="UIT RAG system benchmark")
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--judge-model", default=os.getenv("EVAL_JUDGE_MODEL"))
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--production-mode", action="store_true")
    parser.add_argument("--no-gate", action="store_true")
    return parser.parse_args()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize(text):
    return " ".join((text or "").casefold().split())


def source_lookup():
    lookup = {}
    for name in ("train.csv", "val.csv", "test.csv"):
        path = RAW_DIR / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                key = normalize(row.get("question"))
                source = {
                    "article": row.get("article", ""),
                    "document": row.get("document", ""),
                }
                values = lookup.setdefault(key, [])
                if key and source not in values:
                    values.append(source)
    return lookup


def load_cases(config):
    config_dir = CONFIG_PATH.parent
    core = read_json((config_dir / config["core_dataset"]).resolve())
    lookup = source_lookup()
    cases = []
    missing_sources = []
    for index, item in enumerate(core, 1):
        question = item["question"]
        expected_sources = lookup.get(normalize(question), [])
        if not expected_sources:
            missing_sources.append(question)
        cases.append({
            "id": f"core-{index:03d}",
            "core_index": index,
            "category": "academic_qa",
            "question": question,
            "chat_history": [],
            "ground_truth": item["ground_truth"],
            "expected_behavior": (
                "Trả lời đúng bằng tài liệu truy xuất, không thêm dữ kiện "
                "không có căn cứ."
            ),
            "expected_sources": expected_sources,
            "critical": False,
        })
    if missing_sources:
        raise RuntimeError(
            "Không ánh xạ được expected source cho "
            f"{len(missing_sources)} core cases. Kiểm tra data/raw."
        )
    cases += read_json(config_dir / config["supplemental_dataset"])
    return cases


def select_cases(cases, config, args):
    if args.case_id:
        wanted = set(args.case_id)
        selected = [case for case in cases if case["id"] in wanted]
        missing = wanted - {case["id"] for case in selected}
        if missing:
            raise ValueError(f"Unknown case IDs: {sorted(missing)}")
    elif args.profile == "smoke":
        core = set(config["smoke_core_indices"])
        extra = set(config["smoke_supplemental_ids"])
        selected = [
            case for case in cases
            if case.get("core_index") in core or case["id"] in extra
        ]
    else:
        selected = cases
    return selected[:args.limit] if args.limit else selected


def render_context(documents, max_chars):
    if not documents:
        return "[KHÔNG CÓ TÀI LIỆU]"
    blocks, used = [], 0
    for index, document in enumerate(documents, 1):
        metadata = getattr(document, "metadata", {}) or {}
        header = (
            f"[Tài liệu {index}]\n"
            f"Văn bản: {metadata.get('document', 'Không rõ')}\n"
            f"Điều khoản: {metadata.get('article', 'Không rõ')}\n"
            f"Loại: {metadata.get('type', 'internal')}\n"
        )
        remaining = max_chars - used - len(header)
        if remaining <= 0:
            break
        block = header + (getattr(document, "page_content", "") or "")[:remaining]
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def source_was_retrieved(expected, documents) -> Optional[bool]:
    if not expected:
        return None
    for source in expected:
        article = normalize(source.get("article"))
        document_name = normalize(source.get("document"))
        for item in documents:
            metadata = getattr(item, "metadata", {}) or {}
            article_ok = not article or normalize(metadata.get("article")) == article
            document_ok = (
                not document_name
                or normalize(metadata.get("document")) == document_name
            )
            if article_ok and document_ok:
                return True
    return False


def judge_prompt(case, answer, context):
    history = json.dumps(case.get("chat_history", []), ensure_ascii=False)
    sources = json.dumps(case.get("expected_sources", []), ensure_ascii=False)
    return f"""Bạn là giám khảo độc lập của một RAG chatbot học vụ UIT.

Chấm 0-4 cho từng tiêu chí:
- correctness: đúng ground truth, đúng điều kiện, con số và chương trình.
- faithfulness: mọi khẳng định học vụ được retrieved context hỗ trợ.
- relevance: đúng trọng tâm, rõ ràng, không lan man.
- behavior: đáp ứng expected behavior của loại test.

Ground truth rỗng nghĩa là chấm chủ yếu theo expected behavior. Không phạt cách
diễn đạt khác nếu ý nghĩa tương đương. Với direct/clarification/refusal đúng và
không có dữ kiện học vụ bịa đặt, faithfulness có thể là 4.

critical_failure=true nếu chatbot bịa dữ liệu cá nhân/thời gian thực, trả lời
nội dung phải từ chối, không làm rõ câu cực mơ hồ, hoặc kết luận học vụ quan
trọng trái ground truth.

ID: {case["id"]}
CATEGORY: {case["category"]}
QUESTION: {case["question"]}
CHAT HISTORY: {history}
EXPECTED BEHAVIOR: {case.get("expected_behavior", "")}
GROUND TRUTH: {case.get("ground_truth", "")}
EXPECTED SOURCES: {sources}

RETRIEVED CONTEXT:
{context}

SYSTEM ANSWER:
{answer}
"""


def usage_context():
    try:
        from langchain_community.callbacks.manager import get_openai_callback
        return get_openai_callback()
    except Exception:
        return nullcontext(SimpleNamespace(
            prompt_tokens=0, completion_tokens=0, total_tokens=0, total_cost=0.0
        ))


def create_judge(model):
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

    class Judgement(BaseModel):
        correctness: int = Field(ge=0, le=4)
        faithfulness: int = Field(ge=0, le=4)
        relevance: int = Field(ge=0, le=4)
        behavior: int = Field(ge=0, le=4)
        critical_failure: bool
        reason: str

    return ChatOpenAI(model=model, temperature=0).with_structured_output(Judgement)


def score(judgement, weights):
    value = sum(
        getattr(judgement, name) * weight for name, weight in weights.items()
    )
    return round(value / 4, 4)


def run_case(case, rag_agent, judge, config):
    started = time.perf_counter()
    with usage_context() as usage:
        result = rag_agent.invoke({
            "question": case["question"],
            "chat_history": case.get("chat_history", []),
            "retry_count": 0,
            "confidence_score": 0.0,
            "documents": [],
            "answer": "",
            "standalone_question": "",
            "is_web_searched": False,
            "needs_clarification": False,
            "clarification_question": "",
            "cacheable": True,
        })
        answer = result.get("answer", "")
        documents = result.get("documents", [])
        context = render_context(
            documents, int(os.getenv("EVAL_MAX_CONTEXT_CHARS", "40000"))
        )
        judgement = judge.invoke(judge_prompt(case, answer, context))

    critical = judgement.critical_failure or (
        case.get("critical", False)
        and (judgement.behavior <= 1 or judgement.faithfulness <= 1)
    )
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "answer": answer,
        "ground_truth": case.get("ground_truth", ""),
        "documents_retrieved": len(documents),
        "source_hit": source_was_retrieved(
            case.get("expected_sources", []), documents
        ),
        "web_search_used": bool(result.get("is_web_searched", False)),
        "retry_count": int(result.get("retry_count", 0)),
        "retrieval_confidence": float(result.get("confidence_score", 0)),
        "correctness": judgement.correctness,
        "faithfulness": judgement.faithfulness,
        "relevance": judgement.relevance,
        "behavior": judgement.behavior,
        "overall_score": score(judgement, config["score_weights"]),
        "critical_failure": critical,
        "judge_reason": judgement.reason,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0)),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0)),
        "total_tokens": int(getattr(usage, "total_tokens", 0)),
        "estimated_cost_usd": round(float(getattr(usage, "total_cost", 0)), 6),
    }


def average(values):
    return round(sum(values) / len(values), 4) if values else 0.0


def percentile(values, fraction):
    if not values:
        return 0.0
    values = sorted(values)
    return round(values[round((len(values) - 1) * fraction)], 3)


def summarize(rows, config, profile, judge_model):
    academic = {"academic_qa", "multi_turn"}
    behavior = {"direct", "out_of_domain", "ambiguous", "negative_rejection"}
    source_values = [row["source_hit"] for row in rows if row["source_hit"] is not None]
    metrics = {
        "overall_score": average([row["overall_score"] for row in rows]),
        "academic_qa_score": average([
            row["overall_score"] for row in rows if row["category"] in academic
        ]),
        "behavior_score": average([
            row["overall_score"] for row in rows if row["category"] in behavior
        ]),
        "source_hit_rate": average([float(value) for value in source_values]),
        "critical_failures": sum(row["critical_failure"] for row in rows),
        "mean_latency_seconds": average([row["latency_seconds"] for row in rows]),
        "p95_latency_seconds": percentile(
            [row["latency_seconds"] for row in rows], 0.95
        ),
        "web_search_rate": average([
            float(row["web_search_used"]) for row in rows
        ]),
        "mean_retry_count": average([float(row["retry_count"]) for row in rows]),
        "total_tokens": sum(row["total_tokens"] for row in rows),
        "estimated_cost_usd": round(
            sum(row["estimated_cost_usd"] for row in rows), 6
        ),
    }
    limits = config["gates"]
    gates = {
        "overall_score": metrics["overall_score"] >= limits["overall_score_min"],
        "academic_qa_score": (
            metrics["academic_qa_score"] >= limits["academic_qa_score_min"]
        ),
        "behavior_score": metrics["behavior_score"] >= limits["behavior_score_min"],
        "source_hit_rate": (
            metrics["source_hit_rate"] >= limits["source_hit_rate_min"]
        ),
        "critical_failures": (
            metrics["critical_failures"] <= limits["critical_failures_max"]
        ),
    }
    categories = {}
    for name in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == name]
        categories[name] = {
            "count": len(selected),
            "overall_score": average([row["overall_score"] for row in selected]),
            "critical_failures": sum(row["critical_failure"] for row in selected),
        }
    return {
        "benchmark": config["name"],
        "benchmark_version": config["version"],
        "profile": profile,
        "judge_model": judge_model,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(rows),
        "metrics": metrics,
        "categories": categories,
        "gates": gates,
        "passed": all(gates.values()),
    }


def error_row(case, error):
    return {
        "id": case["id"], "category": case["category"],
        "question": case["question"], "answer": "",
        "ground_truth": case.get("ground_truth", ""),
        "documents_retrieved": 0,
        "source_hit": False if case.get("expected_sources") else None,
        "web_search_used": False, "retry_count": 0,
        "retrieval_confidence": 0.0, "correctness": 0, "faithfulness": 0,
        "relevance": 0, "behavior": 0, "overall_score": 0.0,
        "critical_failure": True, "judge_reason": f"Runtime error: {error}",
        "latency_seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
        "total_tokens": 0, "estimated_cost_usd": 0.0,
    }


def write_reports(output_dir, summary, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"benchmark-{stamp}.json"
    csv_path = output_dir / f"benchmark-{stamp}.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"summary": summary, "cases": rows}, file, ensure_ascii=False, indent=2)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main():
    args = arguments()
    if args.production_mode:
        os.environ["USE_RERANKER"] = "false"
    config = read_json(CONFIG_PATH)
    cases = select_cases(load_cases(config), config, args)
    if not cases:
        raise RuntimeError("Benchmark has no cases.")

    sys.path.insert(0, str(ROOT / "backend"))
    from app.chatbot import app as rag_agent

    model = args.judge_model or config["judge_model"]
    judge = create_judge(model)
    print(f"Running {config['name']} v{config['version']}: {len(cases)} cases")
    rows = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']}: {case['question']}")
        try:
            row = run_case(case, rag_agent, judge, config)
        except Exception as error:
            row = error_row(case, error)
        rows.append(row)
        print(
            f"  score={row['overall_score']:.3f} source={row['source_hit']} "
            f"latency={row['latency_seconds']:.1f}s "
            f"cost_usd={row['estimated_cost_usd']:.4f}"
        )

    summary = summarize(rows, config, args.profile, model)
    json_path, csv_path = write_reports(args.output_dir, summary, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    return 0 if summary["passed"] or args.no_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
