"""Unit tests for the AI Fake News Detector demo pipeline."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import DATA_PATH, build_pipeline, load_dataset, predict, train_model


class TestLoadDataset(unittest.TestCase):
    def test_dataset_loads_and_has_reasonable_size(self):
        texts, labels = load_dataset()
        self.assertGreaterEqual(len(texts), 200)
        self.assertLessEqual(len(texts), 400)
        self.assertEqual(len(texts), len(labels))

    def test_dataset_has_both_classes(self):
        _, labels = load_dataset()
        self.assertIn(0, labels)
        self.assertIn(1, labels)
        # Roughly balanced, not wildly skewed.
        fake_ratio = sum(labels) / len(labels)
        self.assertGreater(fake_ratio, 0.2)
        self.assertLess(fake_ratio, 0.8)

    def test_dataset_file_exists(self):
        self.assertTrue(os.path.exists(DATA_PATH))


class TestBuildPipeline(unittest.TestCase):
    def test_build_pipeline_returns_untrained_pipeline(self):
        pipeline = build_pipeline()
        self.assertIn("tfidf", pipeline.named_steps)
        self.assertIn("clf", pipeline.named_steps)


class TestTrainModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = train_model()

    def test_train_test_split_sizes(self):
        texts, _ = load_dataset()
        self.assertEqual(self.result.n_train + self.result.n_test, len(texts))
        self.assertGreater(self.result.n_train, 0)
        self.assertGreater(self.result.n_test, 0)

    def test_held_out_accuracy_above_threshold(self):
        # Sanity threshold: this toy pipeline should comfortably beat
        # random guessing on the held-out split.
        self.assertGreater(self.result.accuracy, 0.7)


class TestPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = train_model()

    def test_predict_returns_label_and_confidence(self):
        label, confidence = predict(self.result.pipeline, "City council approves new budget for road repairs")
        self.assertIn(label, ("real", "fake"))
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_predict_flags_obvious_fake_headline(self):
        label, confidence = predict(
            self.result.pipeline,
            "SHOCKING: Doctors discover miracle pill that Big Pharma doesn't want you to know!!!",
        )
        self.assertEqual(label, "fake")
        self.assertGreater(confidence, 0.5)

    def test_predict_flags_obvious_real_headline(self):
        label, confidence = predict(
            self.result.pipeline,
            "City council approves budget for road repairs after public hearing",
        )
        self.assertEqual(label, "real")
        self.assertGreater(confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
