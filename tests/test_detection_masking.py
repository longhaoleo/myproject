from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from detection.masking import process_one
from detection.types import FaceDetection


class _FakeDetector:
    def __init__(self, detections: list[FaceDetection]):
        self.detections = detections

    def detect(self, image):
        return self.detections

    def close(self) -> None:
        return None


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((120, 160, 3), 240, dtype=np.uint8)
    cv2.imwrite(str(path), image)


class EyeMaskingTest(unittest.TestCase):
    def test_process_one_masks_eyes_without_covering_whole_face(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "1.png"
            output_path = root / "out.png"
            _write_image(input_path)

            detector = _FakeDetector(
                [
                    FaceDetection(
                        box=(40, 20, 120, 100),
                        landmarks={
                            "left_eye": (62, 52),
                            "right_eye": (98, 52),
                        },
                    )
                ]
            )

            status, boxes = process_one(
                image_path=input_path,
                output_path=output_path,
                detectors=[detector],
                part_scale_by_view={},
                part_offset_by_view={},
                part_offset_mode="ratio",
            )

            self.assertEqual(status, "ok")
            self.assertEqual(len(boxes), 2)
            output = cv2.imread(str(output_path))
            self.assertIsNotNone(output)
            self.assertTrue(np.all(output[52, 62] == 0))
            self.assertTrue(np.all(output[52, 98] == 0))
            self.assertFalse(np.all(output[80, 80] == 0))

    def test_process_one_falls_back_when_first_detector_has_no_eye_landmarks(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "1.png"
            _write_image(input_path)

            face_only = _FakeDetector([FaceDetection(box=(40, 20, 120, 100))])
            with_eyes = _FakeDetector(
                [
                    FaceDetection(
                        box=(40, 20, 120, 100),
                        landmarks={
                            "left_eye": (62, 52),
                            "right_eye": (98, 52),
                        },
                    )
                ]
            )

            status, boxes = process_one(
                image_path=input_path,
                output_path=None,
                detectors=[face_only, with_eyes],
                part_scale_by_view={},
                part_offset_by_view={},
                part_offset_mode="ratio",
            )

            self.assertEqual(status, "ok")
            self.assertEqual(len(boxes), 2)


if __name__ == "__main__":
    unittest.main()
