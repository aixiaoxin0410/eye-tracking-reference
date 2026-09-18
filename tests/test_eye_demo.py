import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
from eye_demo import (
    AffineCalibration,
    Detection,
    GazeSession,
    detect,
    run_demo,
    synthetic_eye,
)


class EyeDemoTests(unittest.TestCase):
    def test_pixel_detection_on_unseen_positions(self):
        rng = np.random.default_rng(314)
        for side in ("left", "right"):
            for target in rng.uniform(0.15, 0.85, (12, 2)):
                image, truth = synthetic_eye(target, side, rng)
                result = detect(image)
                self.assertTrue(result.valid)
                self.assertLess(np.linalg.norm(result.pupil - truth["pupil"]), 0.7)
                self.assertLess(
                    np.max(np.linalg.norm(result.glints - truth["glints"], axis=1)), 0.5
                )

    def test_missing_glint_and_blink_are_invalid(self):
        for kwargs in ({"blink": True}, {"missing_glint": True}):
            image, _ = synthetic_eye(
                (0.5, 0.5), "left", np.random.default_rng(1), **kwargs
            )
            self.assertFalse(detect(image).valid)

    def test_reject_bad_images(self):
        for image in (np.zeros((0, 0)), np.zeros((3, 3, 3)), np.full((2, 2), np.nan)):
            with self.assertRaises(ValueError):
                detect(image)

    def test_affine_fit_and_unseen_predictions(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(50, 4))
        coefficients = rng.normal(size=(5, 2))
        y = np.column_stack([x, np.ones(len(x))]) @ coefficients
        model = AffineCalibration.fit(x, y)
        unseen = rng.normal(size=4)
        np.testing.assert_allclose(
            model.predict(unseen), np.append(unseen, 1) @ coefficients, atol=1e-10
        )

    def test_reject_degenerate_calibration(self):
        with self.assertRaises(ValueError):
            AffineCalibration.fit(np.ones((12, 4)), np.ones((12, 2)))
        with self.assertRaises(ValueError):
            AffineCalibration.fit(
                np.ones((12, 4)), np.random.default_rng(2).normal(size=(12, 2))
            )

    def test_end_to_end_heldout_and_failure_policy(self):
        with tempfile.TemporaryDirectory() as output:
            report = run_demo(Path(output), seed=42)
            self.assertLess(report["metrics"]["heldout_gaze_rmse_normalized"], 0.03)
            self.assertEqual(report["metrics"]["rejected_invalid_frames"], 3)
            self.assertEqual(len(report["heldout_predictions"]), 32)
            for packet in report["invalid_frame_cases"]:
                self.assertFalse(packet["valid"])
                self.assertTrue(packet["held"])
            self.assertTrue((Path(output) / "eye_tracking_demo.png").exists())

    def test_no_fabricated_first_gaze_and_explicit_hold(self):
        session = GazeSession(AffineCalibration(np.zeros((5, 2))))
        invalid = Detection(False)
        packet = session.process(invalid, invalid, 1, 1)
        self.assertFalse(packet["valid"])
        self.assertFalse(packet["held"])
        self.assertIsNone(packet["gaze"])
        valid = Detection(True, np.array([4, 5]), np.array([[1, 2], [3, 4]]))
        first = session.process(valid, valid, 2, 2)
        held = session.process(valid, invalid, 3, 3)
        self.assertTrue(first["valid"])
        self.assertFalse(held["valid"])
        self.assertTrue(held["held"])
        self.assertEqual(first["gaze"], held["gaze"])

    def test_timestamp_contract(self):
        session = GazeSession(AffineCalibration(np.zeros((5, 2))))
        invalid = Detection(False)
        with self.assertRaises(ValueError):
            session.process(invalid, invalid, 1, 2)
        session.process(invalid, invalid, 3, 3)
        with self.assertRaises(ValueError):
            session.process(invalid, invalid, 3, 3)
        with self.assertRaises(ValueError):
            session.process(invalid, invalid, 2, 2)


if __name__ == "__main__":
    unittest.main()
