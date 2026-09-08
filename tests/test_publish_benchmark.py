import importlib.util
import unittest


spec = importlib.util.spec_from_file_location(
    "publish_benchmark", "scripts/publish_benchmark.py"
)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class PublicBenchmarkReportTests(unittest.TestCase):
    def test_sensitive_and_large_fields_are_removed(self):
        case = {field: None for field in publisher.PUBLIC_CASE_FIELDS}
        case.update({
            "id": "core-001",
            "answer": "private generated answer",
            "ground_truth": "private ground truth",
            "judge_reason": "private judge reasoning",
            "retrieved_context": "private corpus text",
        })

        sanitized = publisher.sanitize_case(case)

        self.assertEqual(set(sanitized), set(publisher.PUBLIC_CASE_FIELDS))
        self.assertNotIn("answer", sanitized)
        self.assertNotIn("ground_truth", sanitized)
        self.assertNotIn("judge_reason", sanitized)
        self.assertNotIn("retrieved_context", sanitized)


if __name__ == "__main__":
    unittest.main()
