from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from detection.artifacts import ArtifactConfig
from detection.box_artifacts import _make_part_masks_from_boxes, _save_generation_artifacts
from detection.types import FaceDetection


class BoxArtifactsTest(unittest.TestCase):
    def test_exports_generation_masks_from_detection_boxes(self) -> None:
        det = FaceDetection(
            box=(20, 10, 100, 90),
            landmarks={
                "left_eye": (42, 35),
                "right_eye": (78, 35),
                "nose": (60, 52),
                "mouth_left": (45, 70),
                "mouth_right": (75, 70),
            },
            view_id="1",
        )
        masks = _make_part_masks_from_boxes(
            image_shape=(120, 160),
            detections=[det],
            view_id="1",
            part_scale_by_view={},
            part_offset_by_view={},
            part_offset_mode="ratio",
        )

        self.assertIn("face", masks)
        self.assertIn("nose", masks)
        self.assertIn("mouth", masks)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rel_path = Path("1/术前/1.JPG")
            _save_generation_artifacts(
                output_root=root,
                rel_path=rel_path,
                part_masks=masks,
                image_shape=(120, 160),
                artifact_config=ArtifactConfig(),
            )

            expected = [
                root / "parts" / "face" / "1/术前/1.png",
                root / "parts" / "nose" / "1/术前/1.png",
                root / "parts" / "mouth" / "1/术前/1.png",
                root / "inpaint_mask" / "1/术前/1.png",
                root / "feather_mask" / "1/术前/1.png",
            ]
            for path in expected:
                self.assertTrue(path.exists(), str(path))
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                self.assertIsNotNone(mask)
                self.assertGreater(int(np.count_nonzero(mask)), 0)


if __name__ == "__main__":
    unittest.main()
