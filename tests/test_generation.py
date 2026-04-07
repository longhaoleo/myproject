from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from generation.eval import run_evaluation
from generation.index import build_generation_manifest
from generation.infer import _generator_seed, _normalized_quant_components
from generation.settings import GenerationPaths, InferenceConfig, LoRATrainConfig
from generation.train import _build_training_samples


def _write_image(path: Path, color: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((64, 64, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_mask(path: Path, box: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((64, 64), dtype=np.uint8)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 255
    cv2.imwrite(str(path), mask)


class GenerationPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        root = Path(self.tmp.name)
        self.input_root = root / "input"
        self.sam_root = root / "sam"
        self.gen_root = root / "gen"
        self.paths = GenerationPaths(
            input_root=self.input_root,
            sam_root=self.sam_root,
            model_dir=root / "model",
            generation_root=self.gen_root,
            prepared_root=self.gen_root / "prepared",
            manifest_path=self.gen_root / "summaries" / "manifest.jsonl",
            issues_path=self.gen_root / "summaries" / "issues.jsonl",
            summaries_root=self.gen_root / "summaries",
            depth_root=self.gen_root / "prepared" / "depth",
            inpaint_mask_root=self.sam_root / "inpaint_mask",
            feather_mask_root=self.sam_root / "feather_mask",
            privacy_mask_root=self.gen_root / "prepared" / "eye_privacy_mask",
            sanitized_root=self.gen_root / "prepared" / "sanitized",
            lora_root=self.gen_root / "lora",
            inference_root=self.gen_root / "infer",
            eval_root=self.gen_root / "eval",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _seed_case(self) -> None:
        rel_paths = [
            Path("1/术前/1.png"),
            Path("1/术后/1.png"),
            Path("1/术前/2.png"),
            Path("1/术后/2.png"),
            Path("1/术前/3.png"),
            Path("1/术后/4.png"),
        ]
        part_boxes = {
            "face": (8, 8, 56, 56),
            "nose": (24, 24, 40, 40),
            "mouth": (20, 40, 44, 50),
            "left_eye": (14, 16, 24, 26),
            "right_eye": (34, 16, 44, 26),
        }
        for rel in rel_paths:
            _write_image(self.input_root / rel)
            _write_image(self.paths.sanitized_root / rel, color=120)
            for part_name, box in part_boxes.items():
                _write_mask(self.sam_root / "parts" / part_name / rel.with_suffix(".png"), box)
            _write_mask(self.sam_root / "inpaint_mask" / rel.with_suffix(".png"), (20, 24, 44, 50))
            _write_mask(self.sam_root / "feather_mask" / rel.with_suffix(".png"), (18, 22, 46, 52))

    def test_build_generation_manifest_reads_detection_outputs(self) -> None:
        self._seed_case()
        rows = build_generation_manifest(paths=self.paths)

        self.assertEqual(len(rows), 6)
        self.assertTrue((self.paths.manifest_path).exists())

        rows = [
            json.loads(line)
            for line in self.paths.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(rows[0]["doctor_token"], "dr_style")
        self.assertEqual(rows[0]["view_token"], "view_1")
        self.assertTrue(rows[0]["image_path"].endswith("/prepared/sanitized/1/术前/1.png"))
        self.assertTrue(rows[0]["inpaint_mask_path"].endswith("1/术前/1.png"))
        self.assertTrue(rows[0]["face_mask_path"].endswith("1/术前/1.png"))
        self.assertEqual(rows[0]["available_pre_views"], ["1", "2", "3"])
        self.assertEqual(rows[0]["available_post_views"], ["1", "2", "4"])
        self.assertEqual(rows[0]["paired_views"], ["1", "2"])
        self.assertEqual(rows[0]["paired_view_count"], 2)
        self.assertTrue(rows[0]["is_paired_view"])

        inpaint_mask = cv2.imread(
            str(self.sam_root / "inpaint_mask" / "1/术前/1.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertIsNotNone(inpaint_mask)
        self.assertGreater(int(inpaint_mask[35, 32]), 0)

        face_mask = cv2.imread(
            str(self.sam_root / "parts" / "face" / "1/术前/1.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertIsNotNone(face_mask)
        ys, xs = np.where(face_mask > 0)
        self.assertGreater(xs.size, 0)
        self.assertEqual(int(xs.max()) - int(xs.min()), int(ys.max()) - int(ys.min()))

    def test_evaluate_counts_existing_inference_outputs(self) -> None:
        self._seed_case()

        for rel in (Path("1/术前/1.png"), Path("1/术前/2.png"), Path("1/术前/3.png")):
            composite_path = self.paths.inference_root / "composited" / rel
            composite_path.parent.mkdir(parents=True, exist_ok=True)
            _write_image(composite_path, color=100)

        summary = run_evaluation(paths=self.paths, max_triptychs=10)
        self.assertGreaterEqual(summary["paired_ok"], 2)
        self.assertEqual(summary["paired_view_count"], 2)
        self.assertEqual(summary["unpaired_pre_view_count"], 1)
        self.assertEqual(summary["unpaired_post_view_count"], 1)
        self.assertEqual(summary["inference_ok"], 3)
        self.assertEqual(summary["predicted_view_count"], 3)
        self.assertEqual(summary["triptych_count"], 2)
        self.assertIsNotNone(summary["hard_identity_similarity"])
        self.assertIsNotNone(summary["soft_face_similarity"])
        self.assertTrue((self.paths.eval_root / "triptychs" / Path("1/术前/1.png")).exists())
        self.assertTrue((self.paths.eval_root / "case_sheets" / "1.png").exists())

    def test_inference_quant_components_are_normalized(self) -> None:
        config = InferenceConfig(
            quantize_components=("unet", "controlnet", "unet", "text_encoder"),
        )
        self.assertEqual(
            _normalized_quant_components(config),
            ("unet", "controlnet", "text_encoder"),
        )

    def test_inference_seed_is_stable_by_case_and_view(self) -> None:
        config = InferenceConfig(seed=7, shared_case_seed=True, case_seed_stride=1000)
        row_a = {"case_id": "1", "view_id": "1"}
        row_b = {"case_id": "1", "view_id": "2"}
        self.assertEqual(_generator_seed(row_a, config, 0), _generator_seed(row_a, config, 9))
        self.assertNotEqual(_generator_seed(row_a, config, 0), _generator_seed(row_b, config, 0))

    def test_training_samples_use_masked_inputs_and_unpaired_self_reconstruction(self) -> None:
        self._seed_case()
        rows = build_generation_manifest(paths=self.paths)
        samples = _build_training_samples(rows, self.paths, LoRATrainConfig())
        self.assertEqual(len(samples), 6)
        sample_types = [sample.sample_type for sample in samples]
        self.assertEqual(sample_types.count("paired_head_reference"), 2)
        self.assertEqual(sample_types.count("paired_edit_crop"), 2)
        self.assertEqual(sample_types.count("self_identity_head"), 2)
        for sample in samples:
            self.assertIn("/prepared/sanitized/", sample.condition_image_path)
            self.assertIn("/inpaint_mask/", sample.inpaint_mask_path)
        self.assertTrue(any(sample.condition_image_path == sample.target_image_path for sample in samples if sample.sample_type == "self_identity_head"))


if __name__ == "__main__":
    unittest.main()
