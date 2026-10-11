"""
common/rsna_dataset.py

RSNA Pneumonia Detection Challenge loader, for unsupervised anomaly detection
with pixel-level localisation ground truth.

WHY THIS DATASET REPLACES THE COVID-19 RADIOGRAPHY DATABASE

On that database, a logistic regression fitted to 8x8 thumbnails separated
Normal from Viral Pneumonia at AUROC 0.988, and from the image borders alone
with the lung field masked out at 0.984. Every class probed between 0.82 and
0.99, and no trained model beat the probe on any class. Its classes were
assembled from different source repositories, so the label correlated with
acquisition rather than anatomy.

RSNA is drawn from a single curated collection (NIH ChestX-ray14 images
re-annotated by the RSNA/STR), which should remove most of that. "Should" is
doing work in that sentence: run controls.py here before trusting anything,
exactly as on the previous dataset.

WHAT IS DIFFERENT FROM THE PREVIOUS LOADER

  1. DICOM, not PNG. Files are 1024x1024, 8-bit, MONOCHROME2, so a divide by
     255 is sufficient and no windowing is needed. MONOCHROME1 is handled
     defensively anyway (it means higher value = darker, and needs
     inverting); the loader reports if it ever sees one.

  2. THREE classes, from stage_2_detailed_class_info.csv:
        Normal                          8,851   training distribution
        Lung Opacity                    9,555   pneumonia, WITH boxes
        No Lung Opacity / Not Normal   11,821   abnormal, NO boxes
     The third class is the interesting one. Evaluated as a second pathology
     set it asks whether the detector flags pneumonia specifically or merely
     "not a clean film", which is a question the previous dataset could not
     pose at all.

  3. BOX GROUND TRUTH. Positives carry 1 to 4 boxes (most have 2). These are
     rasterised into a binary mask and returned as `gt_mask`, enabling
     pixel-level evaluation and therefore an actual localisation claim rather
     than an image-level proxy.

  4. TWO DIFFERENT MASKS, deliberately named apart:
        mask     scoring restriction (a lung mask). RSNA ships none, so this
                 is all-ones unless a segmentation is supplied. It is an
                 INPUT to scoring.
        gt_mask  the annotated pneumonia region. It is the TARGET for
                 localisation metrics and must never be used to restrict the
                 score, which would be circular.

  5. PATIENT IDS. `patientId` is unique per image here (26,684 ids for 26,684
     files), so it identifies a study, not a person. Splitting by it prevents
     the same file appearing twice but CANNOT guarantee that one individual
     does not appear under two ids. State that limitation rather than
     claiming patient-level splits.

  6. AN OPTIONAL CACHE. Decoding 26k DICOMs every epoch is wasteful. At
     128x128 the whole set fits in about 440 MB as uint8, so the first epoch
     writes an .npy cache and later epochs read from it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

NORMAL_CLASS = "Normal"
PATHOLOGY_CLASSES = ["Lung Opacity", "No Lung Opacity / Not Normal"]
ALL_CLASSES = [NORMAL_CLASS] + PATHOLOGY_CLASSES

EXPECTED_COUNTS = {
    "Normal": 8851, "Lung Opacity": 9555, "No Lung Opacity / Not Normal": 11821,
}


def _stable_bucket(name: str, buckets: int = 1000) -> int:
    """Deterministic hash into [0, buckets). md5 because Python's hash() is
    salted per process and would give a different split every run."""
    return int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16) % buckets


# --------------------------------------------------------------------------- #
# Index built once from the two CSVs
# --------------------------------------------------------------------------- #

class RSNAIndex:
    """Joins stage_2_train_labels.csv and stage_2_detailed_class_info.csv into
    one record per image: class name plus its list of boxes."""

    def __init__(self, root: str | Path):
        import pandas as pd

        self.root = Path(root).expanduser()
        self.images_dir = self.root / "stage_2_train_images"
        labels_csv = self.root / "stage_2_train_labels.csv"
        detail_csv = self.root / "stage_2_detailed_class_info.csv"

        for p in (self.images_dir, labels_csv, detail_csv):
            if not p.exists():
                raise FileNotFoundError(
                    f"Expected {p}. The Kaggle layout is stage_2_train_images/, "
                    f"stage_2_train_labels.csv and stage_2_detailed_class_info.csv "
                    f"directly under paths.data_root."
                )

        labels = pd.read_csv(labels_csv)
        detail = pd.read_csv(detail_csv).drop_duplicates("patientId")
        self.class_of = dict(zip(detail.patientId, detail["class"]))

        # One entry per patientId; positives accumulate their boxes.
        self.boxes: dict[str, list[tuple[float, float, float, float]]] = {}
        for r in labels.itertuples(index=False):
            self.boxes.setdefault(r.patientId, [])
            if r.Target == 1 and not (isinstance(r.x, float) and np.isnan(r.x)):
                self.boxes[r.patientId].append((float(r.x), float(r.y),
                                                float(r.width), float(r.height)))

        self.by_class: dict[str, list[str]] = {c: [] for c in ALL_CLASSES}
        missing = 0
        for pid, cls in self.class_of.items():
            if not (self.images_dir / f"{pid}.dcm").is_file():
                missing += 1
                continue
            if cls in self.by_class:
                self.by_class[cls].append(pid)
        for c in self.by_class:
            self.by_class[c].sort()          # deterministic order
        self.missing_files = missing

    def counts(self) -> dict[str, int]:
        return {c: len(v) for c, v in self.by_class.items()}


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class RSNADataset(Dataset):
    """One class's images, with box ground truth.

    Returns dicts:
        image    (1, S, S) float in [0, 1]
        mask     (1, S, S) scoring restriction; all ones (no lung masks here)
        gt_mask  (1, S, S) rasterised annotation boxes; all zeros for negatives
        label    class name
        path     patientId
    """

    def __init__(self, index: RSNAIndex, patient_ids: list[str], image_size: int,
                 cache_dir: str | Path | None = None):
        self.index = index
        self.ids = list(patient_ids)
        self.image_size = image_size
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._warned_mono1 = False

    def __len__(self) -> int:
        return len(self.ids)

    # ------------------------------------------------------------------ #

    def _read_dicom(self, pid: str) -> np.ndarray:
        """Returns a float32 array in [0, 1] at the target resolution."""
        if self.cache_dir is not None:
            cpath = self.cache_dir / f"{pid}_{self.image_size}.npy"
            if cpath.is_file():
                return np.load(cpath).astype(np.float32) / 255.0

        import pydicom
        dcm = pydicom.dcmread(self.index.images_dir / f"{pid}.dcm")
        arr = dcm.pixel_array

        # MONOCHROME1 means higher value = darker. Inverting is required or
        # those images are photographic negatives of the rest.
        if getattr(dcm, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            arr = arr.max() - arr
            if not self._warned_mono1:
                print(f"[rsna_dataset] MONOCHROME1 encountered ({pid}); inverting.")
                self._warned_mono1 = True

        # This release is 8-bit; scale generically so 12/16-bit still works.
        arr = arr.astype(np.float32)
        hi = float(arr.max())
        arr = arr / hi if hi > 0 else arr

        t = torch.from_numpy(arr)[None, None]
        t = F.interpolate(t, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False, antialias=True)
        out = t[0, 0].clamp(0, 1).numpy()

        if self.cache_dir is not None:
            np.save(self.cache_dir / f"{pid}_{self.image_size}.npy",
                    (out * 255).astype(np.uint8))
        return out

    def _box_mask(self, pid: str, native: int = 1024) -> np.ndarray:
        """Rasterise the annotation boxes at the target resolution.

        Box coordinates are given in the native 1024 pixel frame, so they are
        scaled by image_size / 1024 rather than assumed to be normalised.
        """
        m = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        s = self.image_size / float(native)
        for (x, y, w, h) in self.index.boxes.get(pid, []):
            x0, y0 = int(round(x * s)), int(round(y * s))
            x1, y1 = int(round((x + w) * s)), int(round((y + h) * s))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(self.image_size, x1), min(self.image_size, y1)
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] = 1.0
        return m

    def __getitem__(self, i: int) -> dict:
        pid = self.ids[i]
        img = torch.from_numpy(self._read_dicom(pid))[None]
        gt = torch.from_numpy(self._box_mask(pid))[None]
        return {
            "image": img,
            "mask": torch.ones_like(img),      # no lung masks in RSNA
            "gt_mask": gt,
            "label": self.index.class_of.get(pid, "?"),
            "path": pid,
        }


# --------------------------------------------------------------------------- #
# Data module
# --------------------------------------------------------------------------- #

class RSNADataModule:
    def __init__(self, cfg: dict):
        d, p = cfg["data"], cfg["paths"]
        self.index = RSNAIndex(p["data_root"])
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
        self.cache_dir = d.get("cache_dir")

        assert abs(sum(self.splits) - 1.0) < 1e-6, "splits must sum to 1.0"
        bad = (set([self.normal_class]) | set(self.pathology_classes)) - set(ALL_CLASSES)
        if bad:
            raise ValueError(f"Unknown class(es) {sorted(bad)}; have {ALL_CLASSES}")
        if self.normal_class in self.pathology_classes:
            raise ValueError("Normal class also listed as a pathology class")

        if d.get("use_masks", False):
            raise ValueError(
                "data.use_masks is true but RSNA ships no lung masks. Set it false, "
                "or supply a segmentation and extend this loader. Note that the "
                "annotation boxes are ground TRUTH and must not be used to restrict "
                "the score."
            )

    # ------------------------------------------------------------------ #

    def validate_layout(self) -> dict[str, int]:
        counts = self.index.counts()
        if self.index.missing_files:
            print(f"[rsna_dataset] {self.index.missing_files} ids in the CSVs have "
                  f"no matching .dcm file and were dropped.")
        for c, n in counts.items():
            exp = EXPECTED_COUNTS.get(c)
            if exp and n != exp:
                print(f"[rsna_dataset] {c}: found {n}, expected {exp} for the "
                      f"stage 2 release (soft check).")
        return counts

    def _split(self, ids: list[str]):
        tr, va, _ = self.splits
        tr_b, va_b = int(tr * 1000), int((tr + va) * 1000)
        a, b, c = [], [], []
        for pid in ids:
            k = _stable_bucket(f"{self.seed}:{pid}")
            (a if k < tr_b else b if k < va_b else c).append(pid)
        return a, b, c

    def _loader(self, ids, shuffle, bs=None, drop_last=False) -> DataLoader:
        ds = RSNADataset(self.index, ids, self.image_size, self.cache_dir)
        return DataLoader(ds, batch_size=bs or self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          drop_last=drop_last,
                          persistent_workers=self.num_workers > 0)

    def normal_loaders(self):
        ids = self.index.by_class[self.normal_class]
        tr, va, te = self._split(ids)
        self._normal_test_ids = te
        return (self._loader(tr, True, drop_last=True),
                self._loader(va, False, self.eval_batch_size),
                self._loader(te, False, self.eval_batch_size))

    def pathology_loaders(self) -> dict[str, DataLoader]:
        return {c: self._loader(self.index.by_class[c], False, self.eval_batch_size)
                for c in self.pathology_classes}

    def normal_control_loaders(self):
        """Held-out Normal halved, for the normal-versus-normal control.
        AUROC between the halves must land near 0.5."""
        if not hasattr(self, "_normal_test_ids"):
            self.normal_loaders()
        idx = list(self._normal_test_ids)
        h = len(idx) // 2
        return (self._loader(idx[:h], False, self.eval_batch_size),
                self._loader(idx[h:2 * h], False, self.eval_batch_size))