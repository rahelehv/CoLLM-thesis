import unittest

from warm_cold.evaluate import ACCEPTANCE, acceptance_checks, canonical_label


class EvaluateTests(unittest.TestCase):
    def test_labels_are_canonicalized(self):
        self.assertEqual(canonical_label("CoLLM-LoRA"), "collm-lora")
        self.assertEqual(canonical_label("CoLLM_MICA"), "collm-mica")
        self.assertEqual(canonical_label(" MF "), "mf")

    def test_mf_gate_checks_all_three_anchors(self):
        expected = ACCEPTANCE["mf"]
        checks = acceptance_checks(
            "MF",
            expected["overall"],
            expected["warm"],
            expected["cold"],
            1e-6,
        )
        self.assertEqual({check["metric"] for check in checks}, {"overall_auc", "warm_auc", "cold_auc"})
        self.assertTrue(all(check["passed"] for check in checks))

    def test_gate_fails_outside_tolerance(self):
        expected = ACCEPTANCE["collm-lora"]
        checks = acceptance_checks(
            "CoLLM-LoRA",
            expected["overall"] + 1e-5,
            None,
            None,
            1e-6,
        )
        self.assertFalse(checks[0]["passed"])

    def test_unknown_label_has_no_gate(self):
        self.assertIsNone(acceptance_checks("something-else", 0.5, 0.5, 0.5, 1e-6))


if __name__ == "__main__":
    unittest.main()
