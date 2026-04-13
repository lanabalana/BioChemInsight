"""
Post-install patch script for third-party packages.

Apply with:
    python scripts/patch_packages.py

Patches applied:
  1. decimer_segmentation/optimized_complete_structure.py
     np.VisibleDeprecationWarning was removed in NumPy 2.0; replace with
     DeprecationWarning.  (Issue 5)

  2. paddlex processors.py (wherever installed)
     The nms() function is called with a 1-D array when there are no detected
     boxes, causing an IndexError.  Guard both call-sites so nms is only invoked
     when boxes.ndim > 1 and boxes.shape[0] > 0.  (Issue 4)
"""

import importlib
import pathlib
import re
import sys


def _patch_file(path: pathlib.Path, replacements: list[tuple[str, str]], label: str) -> None:
    original = path.read_text(encoding="utf-8")
    patched = original
    applied = []
    for old, new in replacements:
        if old in patched:
            patched = patched.replace(old, new)
            applied.append(repr(old[:60]))
        elif new in patched:
            # Already patched – skip silently.
            pass
        else:
            print(f"  WARNING [{label}]: could not find pattern to replace: {repr(old[:60])}")
    if patched != original:
        path.write_text(patched, encoding="utf-8")
        print(f"  PATCHED {path}  ({', '.join(applied)})")
    else:
        print(f"  OK (no changes needed) {path}")


# ---------------------------------------------------------------------------
# Issue 5: decimer_segmentation – np.VisibleDeprecationWarning removed in NumPy 2
# ---------------------------------------------------------------------------
def patch_decimer():
    try:
        import decimer_segmentation
        pkg_dir = pathlib.Path(decimer_segmentation.__file__).parent
    except ImportError:
        print("SKIP: decimer_segmentation not installed")
        return

    target = pkg_dir / "optimized_complete_structure.py"
    if not target.exists():
        print(f"SKIP: {target} not found")
        return

    _patch_file(
        target,
        [
            ("np.VisibleDeprecationWarning", "DeprecationWarning"),
        ],
        "decimer_segmentation",
    )


# ---------------------------------------------------------------------------
# Issue 4: paddlex – nms() called with 1-D / empty boxes causes IndexError
#
# We wrap every bare  `nms(`  call site inside processors.py with a guard:
#
#   Before:
#       keep = nms(boxes, scores, threshold)
#
#   After:
#       keep = nms(boxes, scores, threshold) if boxes.ndim > 1 and boxes.shape[0] > 0 else []
#
# The regex targets the common pattern used in PaddleX's NMS utilities.
# ---------------------------------------------------------------------------
def patch_paddlex():
    try:
        import paddlex
        pkg_dir = pathlib.Path(paddlex.__file__).parent
    except ImportError:
        print("SKIP: paddlex not installed")
        return

    # Locate processors.py – the exact sub-path varies across PaddleX versions.
    candidates = list(pkg_dir.rglob("processors.py"))
    if not candidates:
        print("SKIP: paddlex processors.py not found")
        return

    # Pattern: assignment that calls nms() and hasn't already been guarded.
    # Matches lines like:
    #   keep = nms(boxes, scores, iou_threshold)
    #   keep_idx = nms(boxes, scores, nms_threshold)
    NMS_PATTERN = re.compile(
        r"^(\s*\w+\s*=\s*)(nms\([^)]+\))(\s*)$",
        re.MULTILINE,
    )

    for processors_py in candidates:
        text = processors_py.read_text(encoding="utf-8")
        new_text = NMS_PATTERN.sub(
            r"\1(\2 if boxes.ndim > 1 and boxes.shape[0] > 0 else [])\3",
            text,
        )
        if new_text != text:
            processors_py.write_text(new_text, encoding="utf-8")
            n = len(NMS_PATTERN.findall(text))
            print(f"  PATCHED {processors_py}  ({n} nms() call-site(s) guarded)")
        else:
            print(f"  OK (no changes needed) {processors_py}")


if __name__ == "__main__":
    print("=== Applying post-install patches ===\n")
    print("--- Issue 5: decimer_segmentation np.VisibleDeprecationWarning ---")
    patch_decimer()
    print()
    print("--- Issue 4: paddlex nms() empty-box IndexError guard ---")
    patch_paddlex()
    print("\nDone.")
