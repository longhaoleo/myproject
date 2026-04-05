"""
项目级数据集工具。

目标：
- 统一所有图片树的遍历方式；
- 统一 `病例 -> 阶段 -> 文件名` 的自然排序规则；
- 统一“按病例组抽样”的逻辑。
"""

from __future__ import annotations

from pathlib import Path
import random
import re


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STAGE_ORDER = {"术前": 0, "术后": 1}


def tokenize(text: str) -> tuple[tuple[int, object], ...]:
    """把文本切成“数字块 + 文本块”，供自然排序使用。"""
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
    """生成统一排序键：病例编号 -> 术前/术后 -> 剩余文件路径。"""
    rel = path.relative_to(root)
    person = rel.parts[0] if len(rel.parts) > 0 else ""
    stage = rel.parts[1] if len(rel.parts) > 1 else ""
    tail = "/".join(rel.parts[2:]) if len(rel.parts) > 2 else ""

    person_key = (0, int(person), ()) if person.isdigit() else (1, 0, tokenize(person))
    stage_key = (STAGE_ORDER.get(stage, 2), tokenize(stage))
    return person_key + stage_key + (tokenize(tail), str(rel).lower())


def iter_images(root: Path):
    """递归产出目录下的所有图片文件。"""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def person_group_name(path: Path, root: Path) -> str:
    """返回图片所属病例组名，约定为第一层目录。"""
    rel = path.relative_to(root)
    return rel.parts[0] if rel.parts else ""


def group_sort_key(name: str):
    """组名自然排序：纯数字优先按数值，其余按文本。"""
    return (0, int(name)) if name.isdigit() else (1, name.lower())


def pick_random_groups(
    image_paths: list[Path],
    root: Path,
    random_group_count: int,
    random_seed: int,
) -> tuple[list[Path], list[str], int]:
    """
    从病例组中随机抽样若干组。

    这里按第一层目录抽样，而不是按单张图片抽样，
    目的是保证同一个病例的多张视角图一起进入实验集。
    """
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
