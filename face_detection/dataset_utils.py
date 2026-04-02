"""
数据集遍历与排序工具。

用于按“人编号 -> 术前/术后 -> 文件名”稳定排序，
避免出现 1,10,100 这种纯字符串排序问题。
"""

from pathlib import Path
import re


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STAGE_ORDER = {"术前": 0, "术后": 1}


def tokenize(text: str) -> tuple[tuple[int, object], ...]:
    # 自然排序分词：数字按数值比较，文本按小写比较。
    parts: list[tuple[int, object]] = []
    for token in re.split(r"(\d+)", text):
        if not token:
            continue
        if token.isdigit():
            parts.append((0, int(token)))
        else:
            parts.append((1, token.lower()))
    return tuple(parts)


def sort_key(path: Path, root: Path) -> tuple[object, ...]:
    # 排序优先级：人编号 -> 阶段(术前/术后) -> 剩余文件路径。
    rel = path.relative_to(root)
    person = rel.parts[0] if len(rel.parts) > 0 else ""
    stage = rel.parts[1] if len(rel.parts) > 1 else ""
    tail = "/".join(rel.parts[2:]) if len(rel.parts) > 2 else ""

    person_key = (0, int(person), ()) if person.isdigit() else (1, 0, tokenize(person))
    stage_key = (STAGE_ORDER.get(stage, 2), tokenize(stage))
    return person_key + stage_key + (tokenize(tail), str(rel).lower())


def iter_images(root: Path):
    # 递归遍历所有支持的图片后缀。
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path

