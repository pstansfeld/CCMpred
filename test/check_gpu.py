#!/usr/bin/env python3
"""
Check that the CUDA code path agrees with the CPU one (CI runners have no GPU).

    test/check_gpu.py path/to/ccmpred [alignment.aln]

Runs the alignment (default example/1atzA.aln) for a fixed 30 iterations on GPU 0
and on the CPU, then compares the two contact matrices: their correlation must be
at least 0.9999 and their top-L contacts (|i-j| > 5) must mostly agree.  The two
paths sum in different orders, so they differ by float rounding, not exactly.
"""
import math
import os
import subprocess
import sys
import tempfile


def read_mat(path):
    with open(path) as f:
        return [[float(v) for v in line.split()] for line in f if not line.startswith("#")]


def top_contacts(m, n):
    pairs = sorted(((m[i][j], i, j) for i in range(len(m)) for j in range(i + 6, len(m))),
                   reverse=True)
    return {(i, j) for _, i, j in pairs[:n]}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ccmpred = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    aln = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "example", "1atzA.aln")
    with tempfile.TemporaryDirectory() as tmp:
        mats = {}
        for name, opts in (("gpu", ["-d", "0"]), ("cpu", ["-t", str(os.cpu_count() or 1)])):
            mats[name] = os.path.join(tmp, f"{name}.mat")
            r = subprocess.run([ccmpred, *opts, "-n", "30", "-e", "0", aln, mats[name]],
                               capture_output=True, text=True)
            if r.returncode or name == "gpu" and "using device" not in r.stdout:
                sys.exit(f"{name} run failed:\n{(r.stdout + r.stderr)[-1500:]}")
        g, c = read_mat(mats["gpu"]), read_mat(mats["cpu"])
    x = [v for row in g for v in row]
    y = [v for row in c for v in row]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    r = sxy / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    L = len(g)
    shared = len(top_contacts(g, L) & top_contacts(c, L))
    print(f"GPU vs CPU: max |difference| {max(abs(a - b) for a, b in zip(x, y)):.2e}, "
          f"correlation {r:.6f}, top-L contacts shared {shared}/{L}")
    ok = r >= 0.9999 and shared >= 0.95 * L
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
