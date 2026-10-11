"""
apply_inpaint_patch.py

Applies the two inpainting edits to YOUR train.py, rather than replacing the
file with a copy that may be out of date.

    cd path/to/this/repo/common
    python apply_inpaint_patch.py train.py

Writes train.py.bak first, refuses to proceed unless both anchors are found
exactly once, and checks the result parses before saving. Run it twice and
the second run reports that the edits are already present rather than
duplicating them.

WHAT IT CHANGES

1. Before the epoch loop: computes the pixelwise mean of the training
   normals, used as the mask fill. Zero is pure black, far outside the
   distribution of a radiograph, and a zero-filled input pushes the encoder
   into a regime it never meets on clean images.

2. In the training step: masks the INPUT while leaving the TARGET as the
   original image. Passing the masked image as the target is what produced
   the near-black reconstructions, because reproducing an occlusion is both
   trivially easy, hence the suspiciously low 0.00054 loss, and useless.

Both are behind `training.inpaint.enabled`, so stages 1 to 7 are unaffected.
"""

import ast
import re
import shutil
import sys
from pathlib import Path

BLOCK1 = '''
    # ---- inpainting: fill value, computed once -----------------------------
    ip = cfg["training"].get("inpaint", {}) or {}
    train_mean = None
    if ip.get("enabled"):
        if ip.get("fill") == "dataset_mean":
            s, n = None, 0
            for _b in train_loader:
                _xb = (_b["image"] if isinstance(_b, dict)
                       else (_b[0] if isinstance(_b, (list, tuple)) else _b))
                s = _xb.sum(0) if s is None else s + _xb.sum(0)
                n += _xb.size(0)
            train_mean = (s / n).to(device)
        logger.info(
            f"inpaint: {ip['patch']} px patches, {ip['test_n']} groups, "
            f"scheme {ip.get('scheme', 'disjoint')}, "
            f"fill {'training mean' if train_mean is not None else 'zero'}")
'''

BLOCK2 = '''                if ip.get("enabled", False):
                    m = training_mask(x.shape[0], x.shape[-1], ip["patch"],
                                      ip["test_n"], x.device,
                                      ip.get("scheme", "disjoint"))
                    out = model(apply_mask(x, m, train_mean), sample=True)
                else:
                    out = model(x, sample=True)
                # The TARGET is the original x in both branches.
                losses = objective(x, out, epoch)
'''

OLD2 = '''                out = model(x, sample=True)
                losses = objective(x, out, epoch)
'''


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "train.py")
    src = path.read_text()

    if "train_mean" in src and "apply_mask(x, m" in src:
        print("Both edits are already present; nothing to do.")
        return

    if "from inpaint import" not in src:
        raise SystemExit(
            "No `from inpaint import ...` line found. Add this near the other\n"
            "imports first:\n\n"
            "    from inpaint import training_mask, apply_mask\n")

    # --- edit 2 first, so edit 1's insertion point is not shifted ---
    if OLD2 not in src:
        raise SystemExit(
            "Could not find the training step to replace. Expected exactly:\n\n"
            + OLD2 +
            "\nIf your indentation or argument order differs, make the edit by\n"
            "hand; the script will not guess.")
    if src.count(OLD2) != 1:
        raise SystemExit(f"Found {src.count(OLD2)} copies of the training step; "
                         "expected 1. Edit by hand.")
    src = src.replace(OLD2, BLOCK2)

    # --- edit 1: insert before the objective is constructed ---
    m = re.search(r"^(\s*)objective\s*=\s*VAEObjective\(", src, re.M)
    if not m:
        m = re.search(r"^(\s*)for epoch in range\(", src, re.M)
    if not m:
        raise SystemExit(
            "Could not find an insertion point for the fill-value block. Paste\n"
            "it by hand before the epoch loop:\n" + BLOCK1)
    src = src[:m.start()] + BLOCK1.lstrip("\n") + "\n" + src[m.start():]

    try:
        ast.parse(src)
    except SyntaxError as e:
        raise SystemExit(f"The patched file does not parse: line {e.lineno}, "
                         f"{e.msg}. Nothing was written.")

    shutil.copy2(path, str(path) + ".bak")
    path.write_text(src)
    print(f"patched {path}  (backup at {path}.bak)")
    for k in ("from inpaint import", "train_mean = None", "apply_mask(x, m",
              "losses = objective(x, out, epoch)"):
        print(f"  {'ok ' if k in src else 'MISSING '}{k}")


if __name__ == "__main__":
    main()