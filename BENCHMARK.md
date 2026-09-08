# UIT RAG Chatbot System Benchmark v2

## Mục tiêu

Benchmark này thay thế cách chạy RAGAS cũ bằng một đánh giá tổng quan, có
ground truth, có kiểm tra nguồn, bao phủ hành vi ngoài happy path và có chi phí
có thể kiểm soát.

Thiết kế tham khảo:

- [RAGAS (EACL 2024)](https://aclanthology.org/2024.eacl-demo.16/): đánh giá
  context, faithfulness và answer quality.
- [ARES (NAACL 2024)](https://aclanthology.org/2024.naacl-long.20/): ba trục
  context relevance, answer faithfulness và answer relevance; khuyến nghị hiệu
  chỉnh automated judge bằng một tập human-labelled nhỏ.
- [RAGChecker](https://arxiv.org/abs/2408.08067): đánh giá claim-level và phân
  biệt correctness, completeness với retrieval quality. Benchmark v2 chỉ lấy
  các metric tổng quan; chưa triển khai chẩn đoán claim-level để tiết kiệm.
- [RGB](https://arxiv.org/abs/2309.01431): bổ sung negative rejection và khả
  năng không bị context/noise dẫn tới câu trả lời thiếu căn cứ.

## Thành phần

Tổng cộng 72 cases:

- 50 academic_qa: câu hỏi có ground truth và expected article/document.
- 4 direct: chào hỏi, danh tính và phạm vi bot.
- 5 out_of_domain: câu hỏi phải từ chối, gồm một prompt-injection case.
- 4 ambiguous: câu thiếu chủ thể phải yêu cầu làm rõ.
- 4 negative_rejection: dữ liệu cá nhân/thời gian thực không có trong corpus.
- 5 multi_turn: câu hỏi tiếp nối cần sử dụng chat history.

Expected source của 50 core cases được ánh xạ từ CSV trong data/raw. Runner sẽ
dừng nếu không ánh xạ đủ, tránh báo source-hit sai.

## Cách chấm

Mỗi case chỉ dùng một structured LLM-judge call, trả bốn điểm từ 0 đến 4:

| Metric | Trọng số | Ý nghĩa |
|---|---:|---|
| Correctness | 35% | Đúng ground truth, điều kiện, con số và chương trình |
| Faithfulness | 30% | Các khẳng định được retrieved context hỗ trợ |
| Relevance | 20% | Trả lời đúng trọng tâm |
| Behavior | 15% | Route/từ chối/làm rõ đúng hành vi mong đợi |

Runner còn đo trực tiếp expected-source hit, web-search rate, retry count,
retrieval confidence, mean/p95 latency, token usage, estimated API cost và
critical failures.

## Profiles và cách chạy

Smoke profile có 20 cases đại diện:

    python scripts/evaluate.py --profile smoke --production-mode --no-gate

Full profile chạy đủ 72 cases trước release:

    python scripts/evaluate.py --profile full --production-mode

Chạy case cụ thể:

    python scripts/evaluate.py --case-id core-001 --case-id ood-003 --no-gate

production-mode phải được dùng nếu muốn phản ánh cấu hình Render, vì nó đặt
USE_RERANKER=false trước khi import pipeline.

## Gate ban đầu

Threshold nằm trong data/benchmark/benchmark_config.json:

- Overall score >= 0.75.
- Academic QA score >= 0.72.
- Behavior score >= 0.80.
- Source hit rate >= 0.70.
- Critical failures = 0.

Lần chạy đầu nên dùng no-gate, xem các failure, review thủ công 10-15% mẫu, sau
đó chốt baseline. Không nên hạ threshold chỉ để pipeline hiện tại pass.

## Chi phí

Benchmark thêm đúng một judge call cho mỗi case thay vì gọi riêng từng metric.
Với GPT-4o mini, ngân sách dự phòng:

- Smoke 20 cases: dự phòng 0.03-0.20 USD.
- Full 72 cases: dự phòng 0.08-0.50 USD.

Đây không phải báo giá cố định. Độ dài chunks, retry và web fallback làm chi phí
thay đổi. Report ghi token và estimated cost thực tế. Biến
EVAL_MAX_CONTEXT_CHARS mặc định là 40000; giảm giá trị này sẽ giảm judge input
nhưng có thể cắt mất bằng chứng ở cuối context.

Baseline production ngày 2026-09-08 chạy 72 cases dùng 492,330 tokens, chi phí
ước tính 0.07621 USD, overall score 0.7729 và không đạt gate. Xem phân tích tại
`docs/benchmark-baseline-v2.md`.

## Output

Mỗi run tạo JSON và CSV có timestamp trong reports/system_benchmark. JSON chứa
summary, kết quả gate và chi tiết từng case. Exit code là 1 khi gate không đạt,
trừ khi dùng no-gate, nên runner có thể đưa vào CI sau khi chốt baseline.
