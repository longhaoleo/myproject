from __future__ import annotations

import unittest

from detection.part_boxes import build_part_prompt_boxes
from detection.types import FaceDetection


class PartBoxesTest(unittest.TestCase):
    def test_build_part_prompt_boxes_default_heuristics(self) -> None:
        det = FaceDetection(
            box=(100, 50, 300, 250),
            landmarks={
                "left_eye": (140, 110),
                "right_eye": (220, 110),
                "nose": (180, 150),
                "mouth_left": (150, 190),
                "mouth_right": (210, 190),
            },
            view_id="1",
        )

        boxes = build_part_prompt_boxes(det, image_w=400, image_h=300)

        self.assertEqual(boxes["face"], (100, 50, 300, 250))
        self.assertEqual(boxes["left_eye"], (118, 96, 162, 124))
        self.assertEqual(boxes["right_eye"], (198, 96, 242, 124))
        self.assertEqual(boxes["nose"], (154, 128, 206, 172))
        self.assertEqual(boxes["mouth"], (136, 174, 224, 206))

    def test_build_part_prompt_boxes_honors_view_tweaks(self) -> None:
        det = FaceDetection(
            box=(100, 50, 300, 250),
            landmarks={"nose": (180, 150)},
            view_id="4",
        )
        boxes = build_part_prompt_boxes(
            det,
            image_w=400,
            image_h=300,
            part_scale_by_view={"4": {"nose": (2.0, 1.0)}},
            part_offset_by_view={"4": {"nose": (0.1, 0.0)}},
        )

        self.assertIn("nose", boxes)
        self.assertEqual(boxes["nose"], (148, 128, 252, 172))


if __name__ == "__main__":
    unittest.main()
