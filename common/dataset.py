"""
dataset.py

COVID-19 Radiography Database loader for unsupervised anomaly detection.

DIFFERENCES FROM THE MORPHAEUS REPRODUCTION LOADER, and why:

  1. LUNG MASKS ARE LOADED. The reproduction deliberately ignored them
     because the paper's protocol scores full unmasked images. That protocol
     is a liability on this dataset. Each class was assembled from different
     source repositories, so borders, padding, burned-in markers and contrast
     curves correlate with class. Restricting the anomaly score to the lung
     field removes the easiest of those shortcuts. Masks are returned with
     every sample and applied at scoring time, not at input time: the model
     still sees the whole radiograph, so it cannot be accused of being handed
     a segmentation prior.

  2. SAMPLES ARE DICTS, not bare tensors. Scoring needs the mask and the
     provenance alongside the image, and threading those through as a tuple
     invites positional mistakes.

  3. A NORMAL-VERSUS-NORMAL CONTROL SPLIT is exposed. The held-out normal
     test set can be halved so that AUROC can be computed between two samples
     of the same distribution. That number must come out near 0.5. If it does
     not, something leaks and no pathology result from the same pipeline is
     trustworthy.

  4. DETERMINISTIC, PATIENT-BLIND SPLITTING is done by sorted filename hash
     rather than torch.random_split, so the split is reproducible across
     machines and library versions, and stable if the class gains files.

CAVEAT THAT NO LOADER CAN FIX: this dataset provides no patient identifiers,
so images from the same patient may appear in both train and test. Treat
absolute numbers here as optimistic and reserve strong claims for a dataset
with patient-level splits.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = (".png", ".jpeg", ".jpg")

NORMAL_CLASS = "Normal"
PATHOLOGY_CLASSES = ["COVID", "Lung_Opacity", "Viral Pneumonia"]
ALL_CLASSES = [NORMAL_CLASS] + PATHOLOGY_CLASSES

EXPECTED_COUNTS = {
    "Normal": 10192, "Lung_Opacity": 6012, "COVID": 3616, "Viral Pneumonia": 1345,
}


def _stable_bucket(name: str, buckets: int = 1000) -> int:
    """Deterministic hash of a filename into [0, buckets).

    md5 rather than hash() because Python's hash is salted per process.
    """
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


class CXRClassFolder(Dataset):
    """One class directory: images/ and, optionally, masks/.

    Returns dicts:
        image  (1, H, W) float in [0, 1]
        mask   (1, H, W) float in {0, 1}; all ones when no mask is available
        label  class name
        path   source filename (for tracing an outlier back to a file)
    """

    def __init__(self, class_dir: str | Path, image_size: int,
                 load_masks: bool = True, mask_dilate: int = 0):
        self.class_dir = Path(class_dir)
        self.name = self.class_dir.name
        self.image_size = image_size
        self.load_masks = load_masks
        self.mask_dilate = mask_dilate

        images_dir = self.class_dir / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {images_dir}. Note that 'Viral Pneumonia' contains a "
                f"space, not an underscore, in the official release."
            )
        self.paths = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images in {images_dir}")

        self.masks_dir = self.class_dir / "masks"
        self.has_masks = self.masks_dir.is_dir()

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, path: Path, nearest: bool = False) -> torch.Tensor:
        img = Image.open(path).convert("L")
        t = torch.from_numpy(
            torch.frombuffer(img.tobytes(), dtype=torch.uint8).numpy().copy()
        ).float().view(1, 1, img.size[1], img.size[0]) / 255.0
        mode = "nearest" if nearest else "bilinear"
        kw = {} if nearest else {"align_corners": False, "antialias": True}
        t = F.interpolate(t, size=(self.image_size, self.image_size), mode=mode, **kw)
        return t[0]

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        image = self._load(path).clamp(0.0, 1.0)

        if self.load_masks and self.has_masks:
            mpath = self.masks_dir / path.name
            if mpath.is_file():
                mask = (self._load(mpath, nearest=True) > 0.5).float()
                if self.mask_dilate > 0:
                    k = 2 * self.mask_dilate + 1
                    mask = (F.max_pool2d(mask.unsqueeze(0), k, 1, self.mask_dilate)
                            .squeeze(0))
            else:
                mask = torch.ones_like(image)
        else:
            mask = torch.ones_like(image)

        return {"image": image, "mask": mask, "label": self.name, "path": path.name}


class Subset(Dataset):
    def __init__(self, base: Dataset, indices: list[int]):
        self.base, self.indices = base, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        return self.base[self.indices[i]]


class CXRDataModule:
    def __init__(self, cfg: dict):
        d, p = cfg["data"], cfg["paths"]
        self.root = Path(p["data_root"]).expanduser()
        self.image_size = d["image_size"]
        self.normal_class = d.get("normal_class", NORMAL_CLASS)
        self.pathology_classes = d.get("pathology_classes", PATHOLOGY_CLASSES)
        if isinstance(self.pathology_classes, str):
            self.pathology_classes = [self.pathology_classes]
        self.splits = (d["train_split"], d["val_split"], d["test_split"])
        self.batch_size = cfg["training"]["batch_size"]
        self.eval_batch_size = cfg.get("evaluation", {}).get("batch_size", self.batch_size)
        self.num_workers = d["num_workers"]
        self.seed = cfg["experiment"]["seed"]
        self.load_masks = d.get("use_masks", True)
        self.mask_dilate = d.get("mask_dilate", 0)

        assert abs(sum(self.splits) - 1.0) < 1e-6, "splits must sum to 1.0"
        bad = (set([self.normal_class]) | set(self.pathology_classes)) - set(ALL_CLASSES)
        if bad:
            raise ValueError(f"Unknown class(es) {sorted(bad)}; have {ALL_CLASSES}")
        if self.normal_class in self.pathology_classes:
            raise ValueError("Normal class also listed as pathology")

    # ------------------------------------------------------------------ #

    def _folder(self, cls: str) -> CXRClassFolder:
        return CXRClassFolder(self.root / cls, self.image_size,
                              self.load_masks, self.mask_dilate)

    def validate_layout(self) -> dict[str, int]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        counts, missing, no_masks = {}, [], []
        for cls in ALL_CLASSES:
            try:
                f = self._folder(cls)
            except FileNotFoundError:
                missing.append(cls)
                continue
            counts[cls] = len(f)
            if self.load_masks and not f.has_masks:
                no_masks.append(cls)
        if missing:
            raise FileNotFoundError(f"Missing class directories: {missing}")
        if no_masks:
            raise FileNotFoundError(
                f"use_masks is on but these classes have no masks/ folder: {no_masks}. "
                f"Masked scoring is a deliberate design choice here; set "
                f"data.use_masks: false only if you accept the shortcut risk."
            )
        return counts

    def _split_indices(self, n_items: int, folder: CXRClassFolder):
        """Deterministic three-way split by filename hash."""
        tr, va, _ = self.splits
        tr_b, va_b = int(tr * 1000), int((tr + va) * 1000)
        train, val, test = [], [], []
        for i, p in enumerate(folder.paths):
            b = _stable_bucket(f"{self.seed}:{p.name}")
            (train if b < tr_b else val if b < va_b else test).append(i)
        return train, val, test

    def _loader(self, ds, shuffle: bool, batch_size: int | None = None,
                drop_last: bool = False) -> DataLoader:
        return DataLoader(
            ds, batch_size=batch_size or self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True, drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
        )

    # ------------------------------------------------------------------ #

    def normal_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        folder = self._folder(self.normal_class)
        tr, va, te = self._split_indices(len(folder), folder)
        self._normal_test_indices = te
        self._normal_folder = folder
        return (
            self._loader(Subset(folder, tr), True, drop_last=True),
            self._loader(Subset(folder, va), False, self.eval_batch_size),
            self._loader(Subset(folder, te), False, self.eval_batch_size),
        )

    def pathology_loaders(self) -> dict[str, DataLoader]:
        return {
            cls: self._loader(self._folder(cls), False, self.eval_batch_size)
            for cls in self.pathology_classes
        }

    def normal_control_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Held-out normal test set halved, for the normal-versus-normal control.

        AUROC between these two must come out near 0.5. Anything else means
        the scoring pipeline separates images for reasons unrelated to
        pathology, and every pathology number from the same pipeline inherits
        that problem.
        """
        if not hasattr(self, "_normal_test_indices"):
            self.normal_loaders()
        idx = list(self._normal_test_indices)
        half = len(idx) // 2
        return (
            self._loader(Subset(self._normal_folder, idx[:half]), False, self.eval_batch_size),
            self._loader(Subset(self._normal_folder, idx[half:2 * half]), False, self.eval_batch_size),
        )
