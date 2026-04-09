from __future__ import annotations

import unittest

from detection.postprocess import enforce_single_face
from detection.settings import default_min_confidence_map
from detection.types import FaceDetection


class DetectionPostprocessTest(unittest.TestCase):
    def test_enforce_single_face_raises_threshold_for_multi_face(self) -> None:
        detections = [
            FaceDetection(box=(10, 10, 90, 90), score=0.93),
            FaceDetection(box=(100, 20, 150, 70), score=0.91),
            FaceDetection(box=(180, 30, 220, 70), score=0.48),
        ]

        filtered = enforce_single_face(detections, min_confidence=0.3, adaptive_threshold_step=0.02)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].box, (10, 10, 90, 90))
        self.assertAlmostEqual(float(filtered[0].score or 0.0), 0.93, places=6)

    def test_enforce_single_face_falls_back_to_largest_when_score_missing(self) -> None:
        detections = [
            FaceDetection(box=(20, 20, 60, 60), score=None),
            FaceDetection(box=(0, 0, 120, 120), score=None),
        ]

        filtered = enforce_single_face(detections, min_confidence=0.3)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].box, (0, 0, 120, 120))

    def test_default_thresholds_are_slightly_higher(self) -> None:
        thresholds = default_min_confidence_map()

        self.assertEqual(thresholds["yolov8-face"], 0.3)
        self.assertEqual(thresholds["retinaface"], 0.55)
        self.assertEqual(thresholds["centerface"], 0.75)


if __name__ == "__main__":
    unittest.main()
