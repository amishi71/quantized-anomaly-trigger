"""Fix for the hls4ml C-simulation build on recent macOS / Apple clang (libc++).

hls4ml's bundled ap_types headers forward-declare `std::complex` inside `namespace std`.
libc++ keeps std::complex in the inline namespace std::__1, so the two declarations clash
("reference to 'complex' is ambiguous"). The fix replaces the forward declaration with a real
`#include <complex>`. Edits the installed hls4ml templates (backups saved as *.h.bak); safe to
re-run. Run once with the project venv active, BEFORE converting:

    python ~/Downloads/patch_hls4ml_macos.py
"""
import importlib.util
import re
import shutil
import sys
from pathlib import Path

PATTERN = re.compile(r"namespace\s+std\s*\{\s*template\s*<\s*typename\s+_Tp\s*>\s*class\s+complex\s*;\s*\}")


def patch_tree(root: Path) -> int:
    patched = 0
    for name in ("ap_int_special.h", "ap_fixed_special.h"):
        for p in sorted(root.rglob(name)):
            s = p.read_text()
            new, k = PATTERN.subn("#include <complex>", s)
            if k:
                bak = p.with_name(p.name + ".bak")
                if not bak.exists():
                    shutil.copy(p, bak)
                p.write_text(new)
                patched += 1
                print("patched:", p)
            else:
                print("nothing to patch (already fixed or different layout):", p)
    return patched


if __name__ == "__main__":
    spec = importlib.util.find_spec("hls4ml")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("hls4ml not found. Activate the project venv first.")
    root = Path(list(spec.submodule_search_locations)[0])
    n = patch_tree(root)
    print(f"\n{n} file(s) patched under {root}")
