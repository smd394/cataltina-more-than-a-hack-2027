"""Rebuild the organisers' four baselines with their own code, check them, and score them.

    python solution/rebuild_baseline.py                  # needs data/
    python solution/rebuild_baseline.py --bootstrap 1000

Three questions, answered with counts and scores only:

  1. Does this machine reproduce the organisers? Each rebuilt file is compared to the committed
     one in evaluation/results/ frame by frame, and by SHA-256 with line endings normalised (git
     on Windows may check the committed files out with '\\r\\n').
  2. Is each file a valid submission? The scorer's own validation.
  3. What do they score on the public five minutes? This is a CORRECTNESS check, nothing else:
     EVALUATION.md publishes the public spellF1 of two of them, so matching those numbers proves
     frames, labels and scorer line up. It is never a reason to prefer one version over another.
  4. Does our pipeline's array version of the nearest-player rule (solution/tracking.py) give the
     organisers' answer on every frame? It must, or pipeline v0 is not the baseline.

The rebuilt files go to out/baseline/, so the committed baselines are never overwritten.
"""

import argparse
import os

import numpy as np

from common import (BASELINE_DIR, COMMITTED_RESULTS_DIR, N_FRAMES, PUBLIC_GT_PATH, pm,
                    read_codes, relative, run, sha256_of, write_labels)
from starter import baseline
from tracking import load_competition, nearest_team

# Params
# Public-window spellF1 published in EVALUATION.md ("The public window is not a leaderboard").
PUBLISHED_PUBLIC_SPELL_F1 = {"baseline_smooth50": 0.3256, "baseline_smooth125": 0.4242}
PUBLISHED_TOLERANCE = 0.00005  # EVALUATION.md rounds to four decimals


def compare_to_committed(path, name):
    """(frames that differ or None if no committed file, same SHA-256 after newline normalising)."""
    committed = os.path.join(COMMITTED_RESULTS_DIR, name + ".csv")
    if not os.path.isfile(committed):
        return None, False
    differing = int(np.sum(read_codes(path) != read_codes(committed)))
    same_bytes = sha256_of(path) == sha256_of(committed, normalise_newlines=True)
    return differing, same_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="bootstrap draws for the spellF1 interval (1000 to compare versions)")
    parser.add_argument("--out", default=BASELINE_DIR, help="where to write the rebuilt files")
    args = parser.parse_args()

    codes = baseline.build()  # prints the team names and how often the ball was located
    gt = read_codes(PUBLIC_GT_PATH)

    print("{:<20} {:>6}  {:<22} {:<6} {:>14} {:>9} {:>14}".format(
        "baseline", "window", "same as committed", "valid", "public spellF1", "macroF1",
        "possessions"))
    all_ok = True
    for name, window in baseline.VARIANTS:
        path = write_labels(os.path.join(args.out, name + ".csv"), baseline.smooth(codes, window))
        differing, same_bytes = compare_to_committed(path, name)
        reason = pm.validate_submission(pm.read_label_csv(path), N_FRAMES)
        result = pm.evaluate(gt, read_codes(path), bootstrap_n=args.bootstrap)

        same = ("no committed file" if differing is None else
                "yes, byte-identical" if same_bytes else
                "labels yes, bytes no" if differing == 0 else
                "NO: {} frames differ".format(differing))
        spell = "{:.4f}".format(result["spell_f1_mean"])
        if args.bootstrap:
            spell += " [{:.2f}-{:.2f}]".format(result["spell_bootstrap_low"],
                                              result["spell_bootstrap_high"])
        print("{:<20} {:>6}  {:<22} {:<6} {:>14} {:>9.4f} {:>6} vs {:<4}".format(
            name, "+/-{}".format(window), same, "yes" if reason is None else "NO", spell,
            result["macro_f1"], result["n_pred_spells"], result["n_gt_spells"]))
        if reason is not None:
            print("    invalid: {}".format(reason))

        published = PUBLISHED_PUBLIC_SPELL_F1.get(name)
        if published is not None and abs(result["spell_f1_mean"] - published) > PUBLISHED_TOLERANCE:
            print("    MISMATCH: EVALUATION.md publishes {:.4f} for this file".format(published))
            all_ok = False
        all_ok = all_ok and reason is None and differing == 0

    # Our pipeline's array version of the same rule must agree with the organisers' loop exactly.
    ours = nearest_team(load_competition())
    differing = int(np.sum(ours != codes))
    print("\npipeline v0 nearest-player step vs the organisers' baseline: {}".format(
        "identical on all {} frames".format(len(codes)) if differing == 0
        else "NO -- {} frames differ".format(differing)))
    all_ok = all_ok and differing == 0

    print("\nwrote {}/ (gitignored)".format(relative(args.out)))
    print("possessions = ours vs the public window's, counted inside the public window only.")
    if all_ok:
        print("OK  -- the rebuild matches the organisers' files and their published public scores.")
    else:
        print("CHECK -- something above does not match; do not build on it until it does.")
        raise SystemExit(1)


if __name__ == '__main__':
    run(main)
