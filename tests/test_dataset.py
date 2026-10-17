import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student_performance.csv"

class TestStudentPerformanceDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA_PATH)

    def test_expected_columns_exist(self):
        expected = {
            "student_id", "study_hours_per_day", "attendance_percent",
            "sleep_hours", "assignment_completion_percent",
            "previous_exam_score", "final_exam_score"
        }
        self.assertTrue(expected.issubset(set(self.df.columns)))

    def test_dataset_has_150_rows(self):
        self.assertEqual(len(self.df), 150)

    def test_student_ids_are_unique(self):
        self.assertTrue(self.df["student_id"].is_unique)

    def test_target_has_no_missing_values(self):
        self.assertFalse(self.df["final_exam_score"].isna().any())

if __name__ == "__main__":
    unittest.main()
