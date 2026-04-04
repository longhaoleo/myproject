"""
数据集遍历与排序工具。

用于按“人编号 -> 术前/术后 -> 文件名”稳定排序，
避免出现 1,10,100 这种纯字符串排序问题。
"""

from pathlib import Path
import random
import re


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STAGE_ORDER = {"术前": 0, "术后": 1}


def tokenize(text: str) -> tuple[tuple[int, object], ...]:
    """把字符串拆成“数字/文本”片段，用于自然排序。"""
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
    """生成排序键：人编号 -> 阶段(术前/术后) -> 文件名。"""
    # 排序优先级：人编号 -> 阶段(术前/术后) -> 剩余文件路径。
    rel = path.relative_to(root)
    person = rel.parts[0] if len(rel.parts) > 0 else ""
    stage = rel.parts[1] if len(rel.parts) > 1 else ""
    tail = "/".join(rel.parts[2:]) if len(rel.parts) > 2 else ""

    person_key = (0, int(person), ()) if person.isdigit() else (1, 0, tokenize(person))
    stage_key = (STAGE_ORDER.get(stage, 2), tokenize(stage))
    return person_key + stage_key + (tokenize(tail), str(rel).lower())


def iter_images(root: Path):
    """递归遍历并产出所有图片路径。"""
    # 递归遍历所有支持的图片后缀。
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def person_group_name(path: Path, root: Path) -> str:
    """返回图片所属的“组名”（约定第一层目录）。"""
    # 约定第一层子目录为“组”（例如 deformity/68/... -> 组名 68）。
    rel = path.relative_to(root)
    return rel.parts[0] if rel.parts else ""


def group_sort_key(name: str):
    """组名自然排序键：纯数字按数值排序。"""
    # 组名自然排序：纯数字按数值，其余按文本。
    return (0, int(name)) if name.isdigit() else (1, name.lower())


def pick_random_groups(
    image_paths: list[Path],
    root: Path,
    random_group_count: int,
    random_seed: int,
) -> tuple[list[Path], list[str], int]:
    """从所有组中随机抽样若干组，只返回这些组内的图片。"""
    groups: set[str] = set()
    for path in image_paths:
        group = person_group_name(path, root)
        if group:
            groups.add(group)

    total_groups = len(groups)
    if random_group_count <= 0 or random_group_count >= total_groups:
        all_groups = sorted(groups, key=group_sort_key)
        return image_paths, all_groups, total_groups

    rng = random.Random(random_seed)
    selected_groups = rng.sample(list(groups), k=random_group_count)
    selected_set = set(selected_groups)
    sampled_paths = [p for p in image_paths if person_group_name(p, root) in selected_set]
    selected_groups = sorted(selected_groups, key=group_sort_key)
    return sampled_paths, selected_groups, total_groups
