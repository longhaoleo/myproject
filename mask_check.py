from pathlib import Path

import cv2
import mediapipe as mp
import torch
from facenet_pytorch import MTCNN
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# MediaPipe FaceMesh 左右眼关键点索引。
LEFT_EYE_INDICES = [
    33, 246, 161, 160, 159, 158, 157, 173,
    133, 155, 154, 153, 145, 144, 163, 7,
]
RIGHT_EYE_INDICES = [
    362, 398, 384, 385, 386, 387, 388, 466,
    263, 249, 390, 373, 374, 380, 381, 382,
]

# MTCNN 动态参数。
MTCNN_MIN_PROB = 0.70
MTCNN_PROFILE_THRESHOLD = 0.22
MTCNN_EYE_W_FACE_RATIO = 0.08
MTCNN_EYE_W_DX_RATIO = 0.28
MTCNN_EYE_H_FACE_RATIO = 0.055
MTCNN_PROFILE_W_FACE_RATIO = 0.12
MTCNN_PROFILE_H_FACE_RATIO = 0.075
MTCNN_PAD_W_RATIO = 0.14
MTCNN_PAD_H_RATIO = 0.16


def safe_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    # 裁剪到图像边界内，非法框返回 None。
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def keep_gap_between_two_boxes(box_a, box_b, min_gap: int = 6):
    # 调整左右边界，确保中间留白，不遮鼻梁。
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    if ax1 > bx1:
        ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 = bx1, by1, bx2, by2, ax1, ay1, ax2, ay2

    if ax2 + min_gap <= bx1:
        return (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2)

    mid = (ax2 + bx1) // 2
    new_ax2 = max(ax1, mid - (min_gap // 2))
    new_bx1 = min(bx2, mid + (min_gap - min_gap // 2))
    return (ax1, ay1, new_ax2, ay2), (new_bx1, by1, bx2, by2)


def load_mtcnn_weights_from_model_dir(detector: MTCNN, model_dir: Path, device: torch.device) -> None:
    # 从本地 model 目录加载 MTCNN 权重。
    weight_map = {
        'pnet': model_dir / 'pnet.pt',
        'rnet': model_dir / 'rnet.pt',
        'onet': model_dir / 'onet.pt',
    }
    for net_name, weight_path in weight_map.items():
        if not weight_path.exists():
            print(f'[WARN] MTCNN {net_name} 权重不存在: {weight_path}')
            continue
        state_dict = torch.load(weight_path, map_location=device)
        net = getattr(detector, net_name)
        net.load_state_dict(state_dict)
        net.to(device).eval()


def create_mtcnn_detector(model_dir: Path):
    # 创建 MTCNN 检测器。
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(model_dir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'MTCNN using device: {device}')
    detector = MTCNN(keep_all=True, device=device)
    load_mtcnn_weights_from_model_dir(detector, model_dir, device)
    return detector


def create_face_landmarker(model_dir: Path):
    # 创建 MediaPipe Face Landmarker。
    model_path = model_dir / 'face_landmarker.task'
    if not model_path.exists():
        raise RuntimeError(f'未找到模型文件: {model_path}')

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def boxes_from_landmarks(landmarks, w: int, h: int, flipped: bool):
    # 从单张人脸关键点生成左右眼独立框。
    def build(indices):
        xs = [int((1.0 - landmarks[i].x) * w) if flipped else int(landmarks[i].x * w) for i in indices]
        ys = [int(landmarks[i].y * h) for i in indices]
        return safe_box(min(xs), min(ys), max(xs), max(ys), w, h)

    left_box = build(LEFT_EYE_INDICES)
    right_box = build(RIGHT_EYE_INDICES)

    if left_box and right_box:
        # 侧脸可能出现假眼：中心过近或面积差太大时只留一只。
        l_w = left_box[2] - left_box[0]
        l_h = left_box[3] - left_box[1]
        r_w = right_box[2] - right_box[0]
        r_h = right_box[3] - right_box[1]
        l_area = max(1, l_w * l_h)
        r_area = max(1, r_w * r_h)
        l_cx = (left_box[0] + left_box[2]) / 2.0
        r_cx = (right_box[0] + right_box[2]) / 2.0
        center_dx = abs(l_cx - r_cx)
        overlap_like = center_dx < max(l_w, r_w) * 0.9
        area_ratio = max(l_area, r_area) / max(1, min(l_area, r_area))

        if overlap_like or area_ratio > 2.6:
            return [left_box if l_area >= r_area else right_box]

        box1, box2 = keep_gap_between_two_boxes(left_box, right_box, min_gap=6)
        return [box1, box2]

    if left_box:
        return [left_box]
    if right_box:
        return [right_box]
    return []


def detect_with_landmarker(image, landmarker):
    # MediaPipe 主流程：先原图检测，失败后再尝试镜像图。
    h, w = image.shape[:2]
    max_eye_idx = max(max(LEFT_EYE_INDICES), max(RIGHT_EYE_INDICES))
    boxes = []

    def detect(img, flipped=False):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        for landmarks in result.face_landmarks:
            if len(landmarks) <= max_eye_idx:
                continue
            boxes.extend(boxes_from_landmarks(landmarks, w, h, flipped))

    detect(image, flipped=False)
    if boxes:
        return boxes

    detect(cv2.flip(image, 1), flipped=True)
    return boxes


def pick_best_mtcnn_face(image, mtcnn_detector):
    # 只取置信度最高的人脸，避免多脸干扰。
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = mtcnn_detector.detect(rgb, landmarks=True)
    if boxes is None or len(boxes) == 0:
        return None

    best_idx = -1
    best_prob = float('-inf')
    for idx in range(len(boxes)):
        prob = float(probs[idx]) if probs is not None else 0.0
        if prob > best_prob:
            best_prob = prob
            best_idx = idx

    if best_idx < 0 or best_prob < MTCNN_MIN_PROB:
        return None

    box = boxes[best_idx]
    landmark = landmarks[best_idx] if landmarks is not None else None
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3])), landmark


def mtcnn_landmarks_reliable(landmarks, face_box) -> bool:
    # 过滤明显错误的眼睛关键点，减少乱遮。
    if landmarks is None or len(landmarks) < 2:
        return False

    x1, y1, x2, y2 = face_box
    face_w = max(1.0, x2 - x1)
    face_h = max(1.0, y2 - y1)
    left_eye, right_eye = landmarks[0], landmarks[1]

    for ex, ey in (left_eye, right_eye):
        if ex < x1 - 0.12 * face_w or ex > x2 + 0.12 * face_w:
            return False
        if ey < y1 + 0.03 * face_h or ey > y1 + 0.72 * face_h:
            return False

    eye_dx = abs(float(left_eye[0] - right_eye[0]))
    if eye_dx > 0.95 * face_w:
        return False
    return True


def detect_with_mtcnn(image, mtcnn_detector, force_single_profile_box: bool = False):
    # MTCNN 兜底：输出 1~2 个眼罩框。
    results = []
    picked = pick_best_mtcnn_face(image, mtcnn_detector)
    if picked is None:
        return results

    face_box, landmarks = picked
    fx1, fy1, fx2, fy2 = face_box
    face_w = max(1.0, fx2 - fx1)
    face_h = max(1.0, fy2 - fy1)

    if not mtcnn_landmarks_reliable(landmarks, face_box):
        return results

    left_eye = landmarks[0]
    right_eye = landmarks[1]
    eye_dx = abs(float(left_eye[0] - right_eye[0]))
    is_profile_like = eye_dx < (MTCNN_PROFILE_THRESHOLD * face_w)
    if force_single_profile_box:
        is_profile_like = True

    if is_profile_like:
        if len(landmarks) > 2:
            nose = landmarks[2]
            d_left = (left_eye[0] - nose[0]) ** 2 + (left_eye[1] - nose[1]) ** 2
            d_right = (right_eye[0] - nose[0]) ** 2 + (right_eye[1] - nose[1]) ** 2
            chosen_eye = left_eye if d_left >= d_right else right_eye
        else:
            chosen_eye = left_eye if left_eye[0] < right_eye[0] else right_eye

        half_w = int(face_w * MTCNN_PROFILE_W_FACE_RATIO)
        half_h = int(face_h * MTCNN_PROFILE_H_FACE_RATIO)
        box = safe_box(
            int(chosen_eye[0] - half_w),
            int(chosen_eye[1] - half_h),
            int(chosen_eye[0] + half_w),
            int(chosen_eye[1] + half_h),
            image.shape[1],
            image.shape[0],
        )
        if box is not None:
            results.append(box)
        return results

    half_w = int(max(face_w * MTCNN_EYE_W_FACE_RATIO, eye_dx * MTCNN_EYE_W_DX_RATIO))
    half_h = int(face_h * MTCNN_EYE_H_FACE_RATIO)
    left_box = safe_box(
        int(left_eye[0] - half_w),
        int(left_eye[1] - half_h),
        int(left_eye[0] + half_w),
        int(left_eye[1] + half_h),
        image.shape[1],
        image.shape[0],
    )
    right_box = safe_box(
        int(right_eye[0] - half_w),
        int(right_eye[1] - half_h),
        int(right_eye[0] + half_w),
        int(right_eye[1] + half_h),
        image.shape[1],
        image.shape[0],
    )

    if left_box and right_box:
        box1, box2 = keep_gap_between_two_boxes(left_box, right_box, min_gap=6)
        results.extend([box1, box2])
    elif left_box:
        results.append(left_box)
    elif right_box:
        results.append(right_box)

    return results


def detect_eye_mask_boxes(image, image_path: Path, landmarker, mtcnn_detector):
    # 统一入口：MediaPipe 优先，失败后 MTCNN 兜底。
    boxes = detect_with_landmarker(image, landmarker)
    if boxes:
        return boxes, 'ok_mp'

    # 4/5 号位通常是侧脸，只保留单眼框。
    force_single_profile_box = image_path.stem in {'4', '5'}
    boxes = detect_with_mtcnn(
        image=image,
        mtcnn_detector=mtcnn_detector,
        force_single_profile_box=force_single_profile_box,
    )
    if boxes:
        return boxes, 'ok_mtcnn'

    return [], 'no_face'
