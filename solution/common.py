"""Paths, constants and the frame <-> time conversions every script in solution/ shares.

Nothing here reads match data. It exists so that the numbers that must never drift -- the frame
count, the frame rate, the graded halves, the kickoffs, the label vocabulary -- come from the
organisers' own modules in exactly one place, and so that every file we write lands in out/.

Frame alignment, the one idea that fails silently if it is wrong
----------------------------------------------------------------
Row i of a submission is frame i of the tactical video, counting from 0. Everything else is
converted onto that index before it is used:

    ATD             frame_id 1..N      ->  video frame = frame_id - 1
    Smart Tagging   seconds            ->  video frame = round(seconds * fps)
    tables          video frame        ->  "h:mm:ss" of the video, and the match clock

The match clock runs from each half's kickoff, not from the first graded frame: the second half
is graded from 110808 but kicks off at 113742 (FAQ.md), and the frames in between are all D.

Printing rule
-------------
Scripts here print aggregates only. run() wraps a script's main so that an unexpected error is
reported as its type and our own file and line, never as a message that might quote data.
"""

import hashlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVALUATION_DIR = os.path.join(ROOT, "evaluation")
for _path in (ROOT, EVALUATION_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402

# Params
TEAM = "Cataltina"

N_FRAMES = GAMES[GAME]["n_video_frames"]
FPS = GAMES[GAME]["fps"]
LABELS = pm.LABELS  # H, A, D, X in that order -- the scorer's vocabulary, never retyped here

# Everything we generate. Gitignored: it is derived from the match data, never committed, and
# deleted together with the data when the competition ends (terms, section 8).
OUT_DIR = os.path.join(ROOT, "out")
BASELINE_DIR = os.path.join(OUT_DIR, "baseline")
SUBMISSION_DIR = os.path.join(OUT_DIR, "submission")
TEAM_DIR = os.path.join(OUT_DIR, "team")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, TEAM + ".csv")

COMMITTED_RESULTS_DIR = os.path.join(EVALUATION_DIR, "results")
PUBLIC_GT_PATH = os.path.join(EVALUATION_DIR, "gt", GAME + "_possession_gt_public.csv")
VALIDATOR_PATH = os.path.join(EVALUATION_DIR, "validate_submission.py")

# Names of the periods of the video, in order (period_of returns one of them).
PERIODS = ("before the match", "1st half", "halftime", "2nd half", "after the final whistle")
PRE_KICKOFF = "before kickoff"


# -- the match timeline ------------------------------------------------------------------------

def graded_blocks(game=GAME):
    """[(half, first_frame, stop_frame)] of each graded half, half-open on the right."""
    return [(half, first, last + 1) for half, first, last in half_video_blocks(game)]


def kickoff_frames():
    """{half: video frame of the kickoff}, from the frames published in DATA.md and FAQ.md."""
    return {1: KICKOFF_VIDEO_FRAMES["1st Half"], 2: KICKOFF_VIDEO_FRAMES["2nd Half"]}


def video_time(frame, fps=FPS):
    """'h:mm:ss' of a video frame: its position in the video, counted from the video's start."""
    seconds = int(frame) // fps
    return "{}:{:02d}:{:02d}".format(seconds // 3600, seconds // 60 % 60, seconds % 60)


def period_of(frame, blocks):
    """Which part of the video a frame is in: one of PERIODS. Needs exactly two halves."""
    (_, first_1, stop_1), (_, first_2, stop_2) = blocks
    if frame < first_1:
        return PERIODS[0]
    if frame < stop_1:
        return PERIODS[1]
    if frame < first_2:
        return PERIODS[2]
    if frame < stop_2:
        return PERIODS[3]
    return PERIODS[4]


def match_clock(frame, blocks, kickoffs, fps=FPS):
    """The referee's clock, 'mm:ss', counted from each half's kickoff; 2nd half starts at 45:00.

    Empty outside the graded halves. PRE_KICKOFF on graded frames before a half has kicked off,
    which happens at the start of the second half (graded from 110808, kickoff at 113742).
    """
    for half, first, stop in blocks:
        if first <= frame < stop:
            since = int(frame) - kickoffs[half]
            if since < 0:
                return PRE_KICKOFF
            seconds = since // fps + (45 * 60 if half == 2 else 0)
            return "{:02d}:{:02d}".format(seconds // 60, seconds % 60)
    return ""


# -- submission files --------------------------------------------------------------------------

def write_labels(path, labels):
    """A frame,label CSV with '\\n' line endings on every OS, so equal labels mean equal bytes.

    The organisers' starter writes in text mode, which on Windows produces '\\r\\n' -- valid, but a
    different SHA-256 for the same labels, which would break the reproducibility seal between
    machines.
    """
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def read_codes(path):
    """Label codes 0..3 (H, A, D, X) of a frame,label CSV, via the scorer's own reader."""
    return pm.label_codes(pm.read_label_csv(path))


def sha256_of(path, normalise_newlines=False):
    """Hex SHA-256 of a file. normalise_newlines treats '\\r\\n' as '\\n' (git autocrlf checkouts)."""
    with open(path, "rb") as handle:
        content = handle.read()
    if normalise_newlines:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def relative(path):
    """A path relative to the repo root, for printing. Never prints anything outside the repo."""
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


# -- running a script under the printing rule --------------------------------------------------

def run(main):
    """Call main(); on an error print only its type and where in solution/ it happened.

    OSError messages are kept, because ours are paths ("match data not found: ..."), which is
    what makes a missing-data error fixable. Everything else could be quoting a data value, so its
    message is dropped.
    """
    try:
        return main()
    except KeyboardInterrupt:
        raise
    except Exception as error:  # noqa: BLE001 -- the whole point is to catch everything
        ours = [frame for frame in traceback.extract_tb(error.__traceback__)
                if os.path.abspath(frame.filename).startswith(HERE)]
        where = "{}:{}".format(relative(ours[-1].filename), ours[-1].lineno) if ours else "?"
        print("\nFAILED  {} at {}".format(type(error).__name__, where))
        if isinstance(error, OSError):
            print(str(error))
        sys.exit(1)
