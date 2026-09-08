import importlib.util
import types
import unittest


spec = importlib.util.spec_from_file_location("system_evaluate", "scripts/evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class BenchmarkDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = evaluation.read_json(evaluation.CONFIG_PATH)
        cls.cases = evaluation.load_cases(cls.config)

    def test_full_and_smoke_profiles_have_expected_size(self):
        args = types.SimpleNamespace(case_id=[], profile="smoke", limit=None)
        smoke = evaluation.select_cases(self.cases, self.config, args)
        self.assertEqual(len(self.cases), 72)
        self.assertEqual(len(smoke), 20)

    def test_every_core_case_has_an_expected_source(self):
        core = [case for case in self.cases if case["category"] == "academic_qa"]
        self.assertEqual(len(core), 50)
        self.assertTrue(all(case["expected_sources"] for case in core))

    def test_weighted_score_is_normalized(self):
        judgement = types.SimpleNamespace(
            correctness=4,
            faithfulness=4,
            relevance=4,
            behavior=4,
        )
        self.assertEqual(
            evaluation.score(judgement, self.config["score_weights"]),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
