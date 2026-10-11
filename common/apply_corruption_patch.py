"""
apply_corruption_patch.py  --  adds training-time corruption to train.py.

    cd path/to/this/repo/common
    python apply_corruption_patch.py train.py

Builds on the inpainting patches already applied: run_epoch takes ip and
train_mean, and the training step already branches on ip. This adds a third
keyword, corr, read from training.corruption, and corrupts the INPUT in the
non-inpainting branch while leaving the TARGET clean.

Idempotent, backs up to train.py.bak3, and refuses to write anything that
does not parse.
"""

import ast
import re
import shutil
import sys
from pathlib import Path


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "train.py")
    s = path.read_text()
    if "corrupt(x, corr)" in s:
        print("corruption already patched; nothing to do.")
        return

    # 1. import
    if "from corruption import corrupt" not in s:
        anchor = re.search(r"^from inpaint import .*$", s, re.M)
        if not anchor:
            raise SystemExit("No `from inpaint import` line found. Apply the inpainting "
                             "patches first (apply_inpaint_patch.py, fix_scope.py).")
        s = s[:anchor.end()] + "\nfrom corruption import corrupt                    # noqa: E402" + s[anchor.end():]

    # 2. widen run_epoch
    m = re.search(r"(def run_epoch\(.*?)(ip=None, train_mean=None)(\))", s, re.S)
    if not m:
        raise SystemExit("run_epoch does not take ip/train_mean; run fix_scope.py first.")
    s = s[:m.start(2)] + "ip=None, train_mean=None, corr=None" + s[m.end(2):]

    # 3. corrupt the input in the plain branch; the target stays x
    old = "                    out = model(x, sample=True)\n"
    if s.count(old) != 1:
        raise SystemExit(f"Expected exactly one plain-branch `out = model(x, sample=True)`, "
                         f"found {s.count(old)}. Edit by hand.")
    s = s.replace(old,
        "                    # Denoising: corrupted INPUT, clean TARGET (losses below\n"
        "                    # still compare against x). corrupt() is a no-op unless\n"
        "                    # training.corruption.enabled is set.\n"
        "                    out = model(corrupt(x, corr), sample=True)\n")

    # 4. read the config once in main and forward it everywhere
    if "corr = cfg[" not in s:
        a = re.search(r"^(\s*)ip = cfg\[\"training\"\]\.get\(\"inpaint\", \{\}\) or \{\}\n", s, re.M)
        if not a:
            raise SystemExit("Could not find where ip is read in main().")
        ind = a.group(1)
        s = (s[:a.end()] + f"{ind}corr = cfg[\"training\"].get(\"corruption\", {{}}) or {{}}\n"
             f"{ind}if corr.get(\"enabled\"):\n"
             f"{ind}    logger.info(f\"corruption: {{corr.get('type')}} {{ {{k: v for k, v in corr.items() if k != 'enabled'}} }}\")\n"
             + s[a.end():])
    s = re.sub(r"run_epoch\(((?:[^()]|\([^()]*\))*?)ip=ip, train_mean=train_mean\)",
               lambda mm: f"run_epoch({mm.group(1)}ip=ip, train_mean=train_mean, corr=corr)", s)

    try:
        ast.parse(s)
    except SyntaxError as e:
        raise SystemExit(f"Patched file does not parse: line {e.lineno}, {e.msg}. Nothing written.")
    shutil.copy2(path, str(path) + ".bak3")
    path.write_text(s)
    print(f"patched {path}  (backup at {path}.bak3)")
    for k in ("from corruption import corrupt", "corr=None",
              "corrupt(x, corr)", "corr=corr"):
        print(f"  {'ok ' if k in s else 'MISSING '}{k}")


if __name__ == "__main__":
    main()
