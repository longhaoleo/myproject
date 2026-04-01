from pathlib import Path
import re

import cv2

from mask_apply import draw_black_eye_masks
from mask_check import (
    MTCNN_PAD_H_RATIO,
    MTCNN_PAD_W_RATIO,
    create_face_landmarker,
    create_mtcnn_detector,
    detect_eye_mask_boxes,
)


# 支持的图片后缀
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# 子目录排序：术前在前，术后在后
STAGE_ORDER = {'术前': 0, '术后': 1}

# 模型目录
MODEL_DIR = Path(__file__).resolve().parent / 'model'

# 跳过原因
SKIP_REASON = {
    'read_failed': '读取失败',
    'no_face': '未检测到人脸',
}


def tokenize(text: str) -> tuple[tuple[int, object], ...]:
    # 自然排序分词：数字按数值排序，其他按字符串排序。
    parts: list[tuple[int, object]] = []
    for token in re.split(r'(\d+)', text):
        if not token:
            continue
        parts.append((0, int(token)) if token.isdigit() else (1, token.lower()))
    return tuple(parts)


def sort_key(path: Path, root: Path) -> tuple[object, ...]:
    # 按 person_id -> 术前/术后 -> 文件名 排序。
    rel = path.relative_to(root)
    person = rel.parts[0] if len(rel.parts) > 0 else ''
    stage = rel.parts[1] if len(rel.parts) > 1 else ''
    tail = '/'.join(rel.parts[2:]) if len(rel.parts) > 2 else ''
    person_key = (0, int(person), ()) if person.isdigit() else (1, 0, tokenize(person))
    stage_key = (STAGE_ORDER.get(stage, 2), tokenize(stage))
    return person_key + stage_key + (tokenize(tail), str(rel).lower())


def iter_images(root: Path):
    # 递归扫描所有图片文件。
    for path in root.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def process_one(image_path: Path, output_path: Path, landmarker, mtcnn_detector, pad_x: int, pad_y: int) -> str:
    # 单图处理：先检测眼罩框，再画黑色遮挡块。
    image = cv2.imread(str(image_path))
    if image is None:
        return 'read_failed'

    boxes, source = detect_eye_mask_boxes(
        image=image,
        image_path=image_path,
        landmarker=landmarker,
        mtcnn_detector=mtcnn_detector,
    )
    if not boxes:
        return 'no_face'

    draw_pad_x = pad_x
    draw_pad_y = pad_y

    # MTCNN 框按自身大小做动态扩边。
    if source == 'ok_mtcnn':
        avg_w = sum((x2 - x1) for x1, _, x2, _ in boxes) / max(1, len(boxes))
        avg_h = sum((y2 - y1) for _, y1, _, y2 in boxes) / max(1, len(boxes))
        draw_pad_x = int(avg_w * MTCNN_PAD_W_RATIO)
        draw_pad_y = int(avg_h * MTCNN_PAD_H_RATIO)

    draw_black_eye_masks(
        image=image,
        boxes=boxes,
        pad_x=draw_pad_x,
        pad_y=draw_pad_y,
        min_gap=8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return source


def main():
    # 这里改成你的输入/输出目录。
    input_root = Path('~/datasets/deformity').expanduser().resolve()
    output_root = Path('~/datasets/deformity_masked').expanduser().resolve()

    # MediaPipe 的基础扩边像素。
    padding_x_pixels = 18
    padding_y_pixels = 12

    image_paths = sorted(iter_images(input_root), key=lambda p: sort_key(p, input_root))
    if not image_paths:
        print(f'未找到图片: {input_root}')
        return

    mtcnn_detector = create_mtcnn_detector(MODEL_DIR)

    ok_count = 0
    ok_mp_count = 0
    ok_mtcnn_count = 0
    skip_count = 0

    with create_face_landmarker(MODEL_DIR) as landmarker:
        for image_path in image_paths:
            output_path = output_root / image_path.relative_to(input_root)
            status = process_one(
                image_path=image_path,
                output_path=output_path,
                landmarker=landmarker,
                mtcnn_detector=mtcnn_detector,
                pad_x=padding_x_pixels,
                pad_y=padding_y_pixels,
            )

            if status in {'ok_mp', 'ok_mtcnn'}:
                ok_count += 1
                if status == 'ok_mp':
                    ok_mp_count += 1
                    print(f'已处理(MediaPipe): {output_path}')
                else:
                    ok_mtcnn_count += 1
                    print(f'已处理(MTCNN): {output_path}')
            else:
                skip_count += 1
                print(f'已跳过({SKIP_REASON.get(status, status)}): {image_path}')

    print(
        f'处理完成，总成功 {ok_count} 张 (MediaPipe {ok_mp_count} / MTCNN {ok_mtcnn_count})，'
        f'跳过 {skip_count} 张。'
    )


if __name__ == '__main__':
    main()
