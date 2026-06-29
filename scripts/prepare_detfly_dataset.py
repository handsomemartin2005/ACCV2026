"""Prepare Det-Fly for Ultralytics/RT-DETR training.

The official Det-Fly download contains VOC XML annotations and JPEG images.
This script creates a clean project-local dataset with:
  - images/{train,val,test}
  - labels/{train,val,test} in YOLO txt format
  - annotations/voc/{train,val,test} with cleaned XML filenames
  - annotations/instances_{train,val,test}.json in COCO format
  - dataset YAML for Ultralytics

It intentionally does not modify the BaiduNetdisk download directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


CLASS_NAME = "UAV"
CLASS_ID = 0


@dataclass(frozen=True)
class Record:
    stem: str
    source_group: str
    xml_path: Path
    image_path: Path
    width: int
    height: int
    boxes: tuple[tuple[float, float, float, float], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(r"D:\BaiduNetdiskDownload\Annotations"),
        help="Path to Det-Fly VOC XML annotations.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path(r"D:\BaiduNetdiskDownload\JPEGImages"),
        help="Path to Det-Fly JPEG images.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"D:\PyCharmPojects\Graph-MDETR\dataset\detfly"),
        help="Output dataset root.",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path(r"D:\PyCharmPojects\Graph-MDETR\dataset\detfly.yaml"),
        help="Output Ultralytics dataset YAML.",
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Use NTFS hardlinks for large files when possible, or copy files.",
    )
    return parser.parse_args()


def normalize_stem(stem: str) -> str:
    """Drop BaiduNetdisk duplicate suffixes such as '(1)'."""
    if stem.endswith("(1)"):
        return stem[:-3]
    return stem


def find_image(images_root: Path, group: str, stem: str) -> Path | None:
    group_dir = images_root / group
    candidates = [
        group_dir / f"{stem}.jpg",
        group_dir / f"{stem}.JPG",
        group_dir / f"{stem}.jpeg",
        group_dir / f"{stem}.JPEG",
        group_dir / f"{stem}.jpg.baiduyun.p.downloading",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_record(xml_path: Path, images_root: Path) -> Record | None:
    raw_stem = xml_path.stem
    stem = normalize_stem(raw_stem)
    if raw_stem != stem:
        return None

    group = xml_path.parent.name
    image_path = find_image(images_root, group, stem)
    if image_path is None:
        raise FileNotFoundError(f"Missing image for {xml_path}")

    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")
    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}: {width}x{height}")

    boxes: list[tuple[float, float, float, float]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", CLASS_NAME)
        if name != CLASS_NAME:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        xmin = max(0.0, min(xmin, width - 1.0))
        ymin = max(0.0, min(ymin, height - 1.0))
        xmax = max(0.0, min(xmax, float(width)))
        ymax = max(0.0, min(ymax, float(height)))
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append((xmin, ymin, xmax, ymax))

    return Record(
        stem=stem,
        source_group=group,
        xml_path=xml_path,
        image_path=image_path,
        width=width,
        height=height,
        boxes=tuple(boxes),
    )


def split_records(records: list[Record]) -> dict[str, list[Record]]:
    """Contiguous 80/10/10 split inside each source group."""
    by_group: dict[str, list[Record]] = {}
    for record in records:
        by_group.setdefault(record.source_group, []).append(record)

    splits = {"train": [], "val": [], "test": []}
    for group in sorted(by_group):
        group_records = sorted(by_group[group], key=lambda r: r.stem)
        n = len(group_records)
        train_end = int(n * 0.8)
        val_end = train_end + int(n * 0.1)
        splits["train"].extend(group_records[:train_end])
        splits["val"].extend(group_records[train_end:val_end])
        splits["test"].extend(group_records[val_end:])

    for split in splits:
        splits[split].sort(key=lambda r: r.stem)
    return splits


def ensure_dirs(out: Path) -> None:
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out / "annotations" / "voc" / split).mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    if dst.exists():
        return "exists"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"
    shutil.copy2(src, dst)
    return "copy"


def write_yolo_label(record: Record, label_path: Path) -> None:
    lines = []
    for xmin, ymin, xmax, ymax in record.boxes:
        cx = ((xmin + xmax) / 2.0) / record.width
        cy = ((ymin + ymax) / 2.0) / record.height
        bw = (xmax - xmin) / record.width
        bh = (ymax - ymin) / record.height
        lines.append(f"{CLASS_ID} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_coco(split: str, records: list[Record], out: Path) -> None:
    images = []
    annotations = []
    ann_id = 1
    for image_id, record in enumerate(records, start=1):
        file_name = f"images/{split}/{record.stem}.jpg"
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": record.width,
                "height": record.height,
            }
        )
        for xmin, ymin, xmax, ymax in record.boxes:
            width = xmax - xmin
            height = ymax - ymin
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [xmin, ymin, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": CLASS_NAME, "supercategory": "aircraft"}],
    }
    (out / "annotations" / f"instances_{split}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_yaml(yaml_path: Path, dataset_root: Path) -> None:
    root = dataset_root.resolve().as_posix()
    content = (
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")


def write_readme(out: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Det-Fly prepared dataset",
        "",
        "Prepared from the official Det-Fly BaiduNetdisk download.",
        "Original source folders were not modified.",
        "",
        "Layout:",
        "- `images/{train,val,test}`: cleaned RGB JPEG images",
        "- `labels/{train,val,test}`: YOLO-format labels, one class `UAV`",
        "- `annotations/voc/{train,val,test}`: cleaned VOC XML annotations",
        "- `annotations/instances_{train,val,test}.json`: COCO-format annotations",
        "",
        "Split policy: sorted contiguous 80/10/10 split inside each source group.",
        "",
        "Summary:",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.annotations.exists():
        raise FileNotFoundError(args.annotations)
    if not args.images.exists():
        raise FileNotFoundError(args.images)

    xml_paths = sorted(args.annotations.rglob("*.xml"))
    records: list[Record] = []
    skipped_duplicate_xml = 0
    for xml_path in xml_paths:
        record = read_record(xml_path, args.images)
        if record is None:
            skipped_duplicate_xml += 1
            continue
        records.append(record)

    seen: set[str] = set()
    unique_records: list[Record] = []
    for record in sorted(records, key=lambda r: r.stem):
        if record.stem in seen:
            raise ValueError(f"Duplicate canonical stem: {record.stem}")
        seen.add(record.stem)
        unique_records.append(record)

    splits = split_records(unique_records)
    ensure_dirs(args.out)

    link_stats = {"hardlink": 0, "copy": 0, "exists": 0}
    total_boxes = 0
    for split, split_records_ in splits.items():
        for record in split_records_:
            image_dst = args.out / "images" / split / f"{record.stem}.jpg"
            xml_dst = args.out / "annotations" / "voc" / split / f"{record.stem}.xml"
            label_dst = args.out / "labels" / split / f"{record.stem}.txt"
            link_stats[link_or_copy(record.image_path, image_dst, args.mode)] += 1
            link_stats[link_or_copy(record.xml_path, xml_dst, args.mode)] += 1
            write_yolo_label(record, label_dst)
            total_boxes += len(record.boxes)
        write_coco(split, split_records_, args.out)

    (args.out / "classes.txt").write_text(f"{CLASS_NAME}\n", encoding="utf-8")
    write_yaml(args.yaml, args.out)

    summary = {
        "records": len(unique_records),
        "boxes": total_boxes,
        "skipped_duplicate_xml": skipped_duplicate_xml,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "hardlinked_or_copied_images_and_xml": link_stats,
        "source_annotations": str(args.annotations),
        "source_images": str(args.images),
        "dataset_yaml": str(args.yaml),
    }
    write_readme(args.out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
