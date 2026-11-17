"""Check the seals: what must never change silently in the submission and the code that makes it.
Prints OK / FAIL per seal, counts only.

    python solution/check_seals.py out/submission/Cataltina.csv    # seals 1-4 on a file, 6 always
    python solution/check_seals.py --self-test                      # the checker checks itself
    python solution/check_seals.py --rebuild                        # seal 5: build twice, compare
    python solution/check_seals.py --rebuild --python <py311>\\python.exe   # ship: other Python

Seals checked here:

  1  exactly the video's frame count of rows (from games.py), and no retyped copy of it in code
  2  frame column 0..n-1 in order
  3  labels only H, A, D, X, and no retyped label list in code
  4  the official validator, run as a separate process on this exact path, exits 0; the file's
     SHA-256 is printed so the uploaded copy can be compared to it
  5  reproducible: solution/pipeline.py run twice, in separate processes, writes two files with
     the same SHA-256 (and the same as out/submission/Cataltina.csv, if it exists). --python runs
     the second build under another interpreter, e.g. a clean Python 3.11 venv at ship time
  6  privacy: the match-derived paths (data/, out/, external/, .venv-vision/) are ignored by git,
     a normal source path is not, nothing under out/, data/ or external/ is tracked but
     data/README.md, and the pre-commit hook is installed. A copy of this repository may list
     more paths that must be ignored, one a line, in EXTRA_IGNORED_FILE beside this file; they
     are checked only if that file exists

--self-test runs seals 1-4 on the organisers' committed baseline_smooth50.csv (must pass: the
positive control proving the checker can see a valid file), then on a copy with its last row cut
(must fail: proof the checker can fail). Neither needs the match data.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

from common import (COMMITTED_RESULTS_DIR, HERE, LABELS, N_FRAMES, ROOT, SUBMISSION_PATH,
                    VALIDATOR_PATH, pm, relative, run, sha256_of)

# Params
RETYPED_FRAME_COUNT = re.compile(r"199[,_]?319")
RETYPED_LABEL_LIST = re.compile(r"""["']H["'],\s*["']A["'],\s*["']D["']""")
MUST_BE_IGNORED = ("data/x", "out/x", "external/x", ".venv-vision/x")  # in every clone
MUST_NOT_BE_IGNORED = "solution/x.py"  # positive control: check-ignore can say "no"
# More paths this copy of the repository must ignore, one a line ('#' starts a comment). Read
# only if the file exists; a copy without it checks MUST_BE_IGNORED alone.
EXTRA_IGNORED_FILE = os.path.join(HERE, "check_seals_extra.txt")
SELF_TEST_FILE = os.path.join(COMMITTED_RESULTS_DIR, "baseline_smooth50.csv")
PIPELINE_PATH = os.path.join(HERE, "pipeline.py")


def rebuild_seal(python=None, data_dir=None, reference=SUBMISSION_PATH):
    """(seal, ok, detail): two pipeline builds in separate processes give the same SHA-256.

    The second build uses `python` when given, so the same check proves the file is identical
    under another interpreter. The pipeline's own output (counts only) is not echoed.
    """
    hashes = []
    with tempfile.TemporaryDirectory() as folder:
        for index, interpreter in enumerate((sys.executable, python or sys.executable)):
            out = os.path.join(folder, "build{}.csv".format(index + 1))
            command = [interpreter, PIPELINE_PATH, "--out", out]
            if data_dir:
                command += ["--data-dir", data_dir]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    cwd=ROOT)
            if result.returncode != 0 or not os.path.isfile(out):
                return ("5 reproducible", False, "build {} failed (exit {}); run "
                        "solution/pipeline.py to see why".format(index + 1, result.returncode))
            hashes.append(sha256_of(out))
    same = hashes[0] == hashes[1]
    detail = "build 1 {} · build 2 {}{}".format(
        hashes[0][:16], hashes[1][:16], " (other Python)" if python else "")
    if os.path.isfile(reference):
        matches = sha256_of(reference) == hashes[0]
        same = same and matches
        detail += " · {} {}".format(relative(reference), "same" if matches else "DIFFERS")
    return ("5 reproducible", same, detail)


def file_seals(path, n_expected=N_FRAMES):
    """[(seal, ok, detail)] for seals 1-4 on one submission file. Details are counts only."""
    table = pm.read_label_csv(path)
    frames = table["frame"].values
    cleaned = {str(value).strip().upper() for value in table["label"].values}

    result = subprocess.run([sys.executable, VALIDATOR_PATH, path, "--rows", str(n_expected)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=ROOT)
    order_ok = (len(frames) == n_expected and np.issubdtype(frames.dtype, np.integer)
                and np.array_equal(frames, np.arange(n_expected)))
    return [
        ("1 rows", len(table) == n_expected, "{} rows, expected {}".format(len(table), n_expected)),
        ("2 frame order", order_ok, "frame column is 0..{} in order: {}".format(
            n_expected - 1, "yes" if order_ok else "no")),
        ("3 labels", cleaned <= set(LABELS), "labels used: {}".format(
            " ".join(sorted(cleaned)) if len(cleaned) <= len(LABELS) else
            "{} distinct values".format(len(cleaned)))),
        ("4 validator", result.returncode == 0, "official validator exit {} · SHA-256 {}".format(
            result.returncode, sha256_of(path))),
    ]


def code_seals(root=HERE):
    """[(seal, ok, detail)]: no retyped frame count or label list in the *.py files under root."""
    hits = {"1 rows (code)": [], "3 labels (code)": []}
    for folder, _, names in os.walk(root):
        for name in sorted(names):
            if not name.endswith(".py"):
                continue
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8") as handle:
                for number, line in enumerate(handle, start=1):
                    if RETYPED_FRAME_COUNT.search(line):
                        hits["1 rows (code)"].append("{}:{}".format(relative(path), number))
                    if RETYPED_LABEL_LIST.search(line):
                        hits["3 labels (code)"].append("{}:{}".format(relative(path), number))
    return [(seal, not found, "{} retyped copies{}".format(
        len(found), (": " + ", ".join(found)) if found else ""))
        for seal, found in hits.items()]


def must_be_ignored(extra_file=EXTRA_IGNORED_FILE):
    """MUST_BE_IGNORED, then the paths listed in extra_file if it exists (one a line; blank
    lines and lines starting with '#' skipped)."""
    paths = list(MUST_BE_IGNORED)
    if os.path.isfile(extra_file):
        with open(extra_file, encoding="utf-8") as handle:
            paths += [line.strip() for line in handle
                      if line.strip() and not line.strip().startswith("#")]
    return paths


def privacy_seal(root=ROOT, extra_file=EXTRA_IGNORED_FILE):
    """(seal, ok, detail): every must_be_ignored path ignored, the control not, nothing tracked
    under out/, data/ or external/ but data/README.md, the pre-commit hook installed. root is
    the repository to check (the tests pass a throw-away one)."""
    def ignored(path):
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=root).returncode == 0

    paths = must_be_ignored(extra_file)
    missed = [path for path in paths if not ignored(path)]
    control_ok = not ignored(MUST_NOT_BE_IGNORED)
    tracked = subprocess.run(["git", "ls-files", "out", "data", "external"], cwd=root,
                             stdout=subprocess.PIPE, universal_newlines=True).stdout.split()
    leaked = [path for path in tracked if path != "data/README.md"]
    hook = hook_installed(root)
    ok = not missed and control_ok and not leaked and hook == "installed"
    return ("6 privacy", ok, "{}/{} ignored, control {}, tracked under out/ data/: {}, "
            "pre-commit hook {}".format(
                len(paths) - len(missed), len(paths),
                "ok" if control_ok else "FAILED", ", ".join(leaked) or "none besides data/README.md",
                hook))


def hook_installed(root=ROOT):
    """'installed' if root's .git/hooks/pre-commit is our solution/hooks/pre-commit, else what is
    wrong."""
    ours = os.path.join(HERE, "hooks", "pre-commit")
    live = os.path.join(root, ".git", "hooks", "pre-commit")
    if not os.path.isfile(live):
        return "MISSING (cp solution/hooks/pre-commit .git/hooks/pre-commit)"
    with open(ours, "rb") as a, open(live, "rb") as b:
        same = a.read().replace(b"\r\n", b"\n") == b.read().replace(b"\r\n", b"\n")
    return "installed" if same else "OUTDATED (copy solution/hooks/pre-commit again)"


def report(rows):
    """Print the table; return True if every row passed."""
    for seal, ok, detail in rows:
        print("  {:<4} {:<18} {}".format("OK" if ok else "FAIL", seal, detail))
    return all(ok for _, ok, _ in rows)


def self_test():
    """Positive control on the committed baseline, negative control on a truncated copy."""
    print("positive control: {}".format(relative(SELF_TEST_FILE)))
    positive = report(file_seals(SELF_TEST_FILE))

    with open(SELF_TEST_FILE, "rb") as handle:
        lines = handle.read().splitlines(True)
    handle, broken = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(handle, "wb") as out:
            out.writelines(lines[:-1])
        print("\nnegative control: the same file with its last row cut (must FAIL)")
        negative_failed = not report(file_seals(broken))
    finally:
        os.remove(broken)

    ok = positive and negative_failed
    print("\nself-test {}".format("OK: the checker passes a valid file and fails a broken one"
                                  if ok else "FAILED: do not trust this checker until it passes"))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission", nargs="?", help="the frame,label CSV to check")
    parser.add_argument("--self-test", action="store_true", help="run the checker's own controls")
    parser.add_argument("--rebuild", action="store_true",
                        help="seal 5: build twice and compare SHA-256 (needs the data)")
    parser.add_argument("--python", help="with --rebuild: interpreter for the second build")
    parser.add_argument("--data-dir", help="with --rebuild: the data bundle, if not in data/")
    args = parser.parse_args()

    ok = True
    if args.self_test:
        ok = self_test() and ok
        print("")
    if args.rebuild:
        print("rebuild seal (two builds of solution/pipeline.py)")
        ok = report([rebuild_seal(args.python, args.data_dir)]) and ok
        print("")
    if args.submission:
        if not os.path.isfile(args.submission):
            raise IOError("no such file: {}".format(args.submission))
        print("file seals: {}".format(relative(os.path.abspath(args.submission))))
        ok = report(file_seals(args.submission)) and ok
        print("")
    print("code and privacy seals")
    ok = report(code_seals() + [privacy_seal()]) and ok
    print("\n{}".format("ALL OK" if ok else "SEAL FAILURE: fix before committing"))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    run(main)
