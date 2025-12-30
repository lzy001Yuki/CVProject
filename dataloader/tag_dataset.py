import os
import csv
from typing import Optional, Dict, Tuple, List

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TouchAndGoPairDataset(Dataset):
    """
    Dataset for paired RGB (vision) and GelSight (tactile) frames using the
    Touch-and-Go list format.

    Expected list files (e.g., train.txt/test.txt/pretrain.txt) contain lines:
        raw_rel_path, label
    where raw_rel_path is a path that encodes the clip/frame id, and files are
    located under dataroot as:
        <dataroot>/<raw_rel_path[:16]>/video_frame/<basename(raw_rel_path)>
        <dataroot>/<raw_rel_path[:16]>/gelsight_frame/<basename(raw_rel_path)>

    Args:
        list_dir: Directory containing list files.
        dataroot: Root folder for the dataset content.
        split: One of {"train","test","pretrain"}.
        label: One of {"full","rough","hard"}. Controls label remapping for some tasks.
        img_transform: Transform for the RGB image (vision).
        gel_transform: Transform for the GelSight image (tactile).
        captions: Optional mapping from raw_rel_path to a text caption.
    """

    def __init__(
        self,
        list_dir: str,
        dataroot: str,
        split: str = "train",
        label: str = "full",
        img_transform: Optional[transforms.Compose] = None,
        gel_transform: Optional[transforms.Compose] = None,
        captions: Optional[Dict[str, str]] = None,
    ) -> None:
        assert split in {"train", "test", "pretrain"}
        assert label in {"full", "rough", "hard"}
        self.list_dir = list_dir
        self.dataroot = dataroot
        self.split = split
        self.label = label
        self.img_transform = img_transform
        self.gel_transform = gel_transform
        self.captions = captions or {}

        list_name = {
            "train": "train.txt",
            "test": "test.txt",
            "pretrain": "pretrain.txt",
        }[split]
        list_path = os.path.join(list_dir, list_name)
        with open(list_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        if label == "rough":
            alt = {
                "train": "train_rough.txt",
                "test": "test_rough.txt",
            }.get(split)
            if alt:
                path_alt = os.path.join(list_dir, alt)
                if os.path.isfile(path_alt):
                    with open(path_alt, "r") as f:
                        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        self.entries: List[Tuple[str, int]] = []
        for ln in lines:
            # print(f"processing line {ln}")
            raw, target = ln.split(",")
            target = int(target)
            if label == "hard":
                # Replicate the mapping used in the original dataset wrapper
                if target in {7, 8, 9, 11, 13}:
                    target = 1
                else:
                    target = 0
            self.entries.append((raw, target))

    def __len__(self) -> int:
        return len(self.entries)

    def get_labels_tensor(self) -> torch.Tensor:
        """Return all labels as a LongTensor without iterating __getitem__.
        Uses the internal `entries` list for O(N) extraction.
        """
        return torch.as_tensor([t for _, t in self.entries], dtype=torch.long)

    def get_class_counts(self, n_classes: int) -> torch.Tensor:
        """Return bincount of labels with given `n_classes`.
        Fast path avoids per-sample transforms.
        """
        labels = self.get_labels_tensor()
        return torch.bincount(labels, minlength=n_classes)

    def _paths_from_raw(self, raw: str) -> Tuple[str, str]:
        idx = os.path.basename(raw)
        dir_rel = raw[:16]
        base_dir = os.path.join(self.dataroot, dir_rel)
        img_path = os.path.join(base_dir, "video_frame", idx)
        gel_path = os.path.join(base_dir, "gelsight_frame", idx)
        return img_path, gel_path

    def __getitem__(self, index: int):
        raw, target = self.entries[index]
        img_path, gel_path = self._paths_from_raw(raw)

        img = Image.open(img_path).convert("RGB")
        gel = Image.open(gel_path).convert("RGB")

        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gel_transform is not None:
            gel = self.gel_transform(gel)

        caption = self.captions.get(raw)
        return {
            "vision": img,
            "tactile": gel,
            "label": target,
            "raw": raw,
            "caption": caption,
        }


def load_captions_csv(path: str) -> Dict[str, str]:
    """Load a CSV/TSV mapping raw_rel_path -> caption.

    Accepts files with header or headerless, delimiter auto-detected (comma or tab).
    Columns: raw, caption
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    # Try comma first, then tab
    for delim in [",", "\t"]:
        try:
            out: Dict[str, str] = {}
            with open(path, "r") as f:
                reader = csv.reader(f, delimiter=delim)
                rows = list(reader)
            # Drop header if present (heuristic)
            if rows and rows[0] and rows[0][0].lower() in {"raw", "id", "key"}:
                rows = rows[1:]
            for r in rows:
                if not r:
                    continue
                key = r[0].strip()
                cap = r[1].strip() if len(r) > 1 else ""
                if key:
                    out[key] = cap
            return out
        except Exception:
            continue
    raise RuntimeError(f"Failed to parse captions file: {path}")
