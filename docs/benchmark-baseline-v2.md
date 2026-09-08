# System Benchmark v2 — Production Baseline

Ngày chạy: 2026-09-08

Profile: `full` (72 cases)

Runtime: production mode, local reranker disabled

| Metric | Result | Gate |
|---|---:|---|
| Overall score | 0.7729 | Pass |
| Academic QA score | 0.7223 | Pass |
| Behavior score | 0.9368 | Pass |
| Source hit rate | 0.6182 | Fail |
| Critical failures | 11 | Fail |
| Mean latency | 8.1463 s | Observed |
| P95 latency | 13.823 s | Observed |
| Web search rate | 0.2083 | Observed |
| Estimated cost | USD 0.07621 | Observed |

## Category results

| Category | Cases | Score | Critical failures |
|---|---:|---:|---:|
| Academic QA | 50 | 0.7403 | 9 |
| Multi-turn | 5 | 0.5425 | 2 |
| Direct | 4 | 0.7531 | 0 |
| Ambiguous | 4 | 1.0000 | 0 |
| Out-of-domain | 5 | 1.0000 | 0 |
| Negative rejection | 4 | 0.9781 | 0 |

The regression gate failed because source hit was below 70% and critical
failures were above zero. The main observed failure modes were valid academic
queries rejected by the router, missing expected sources, and loss of context
in short multi-turn follow-ups.

Generated per-case JSON/CSV reports are intentionally excluded from Git because
they may contain model outputs and retrieved corpus text.
