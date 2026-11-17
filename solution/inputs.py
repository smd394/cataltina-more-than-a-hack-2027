"""Readers for the match inputs: ball visibility, the club names and the Smart Tagging.

They lean on the organisers' starter code for paths and parsing, and add only the conversion onto
the submission's frame index. No function here prints anything; callers print aggregates.
"""

import difflib
import re
import unicodedata

import numpy as np

from common import FPS, GAME, N_FRAMES, video_time
from games import require
from starter.baseline import team_names
from starter.load_smart_tagging import load as load_smart_tagging

# Params
# Raw ATD line: "<frame_id>:<player>;<player>;...:<ball>", a missing observation written "NaN".
FIELD_SEP = ":"
MISSING = "N"

# A Smart Tagging team value is matched to the home or away club name from the ATD metadata by
# string similarity, because the two files spell clubs differently ("Atletic Club" in the tags).
# Below this similarity to both clubs, the tag is left unassigned rather than guessed.
MIN_TEAM_SIMILARITY = 0.5
UNASSIGNED = "N/A"


def ball_positions(raw_path, n_frames=N_FRAMES):
    """(n_frames, 2) normalised ball x, y per video frame; NaN where the ball was not located.

    The ATD counts frames from 1 and the submission from 0, so record frame_id k lands on video
    frame k - 1. Lines that are not three ':'-separated fields are skipped, as the baseline does.
    Only the ball field is parsed, so this is much faster than reading the players too.
    """
    ball = np.full((n_frames, 2), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            values = parts[2].strip().split(",")
            if len(values) >= 2 and values[0] and values[0][0] != MISSING:
                ball[int(parts[0]) - 1] = (float(values[0]), float(values[1]))
    return ball


def ball_seen(raw_path, n_frames=N_FRAMES):
    """Boolean array over video frames: True where the tracking located the ball."""
    return np.isfinite(ball_positions(raw_path, n_frames)[:, 0])


def load_ball(game=GAME, data_dir=None):
    """ball_positions() on the tactical ATD of a game."""
    return ball_positions(require(["atd_raw"], game, data_dir)[0])


def ball_speed_per_second(ball, pitch, fps=FPS):
    """Median ball speed in m/s for each second of video, NaN where it cannot be measured.

    Speed is measured between consecutive frames where the ball was located on both; a second
    with no such pair has no speed. The median, not the mean, so one tracking jump (the ball
    "teleporting" to a wrong blob) does not make a still ball look fast.
    """
    length, width = pitch
    step = np.full(len(ball), np.nan)
    step[1:] = np.hypot((ball[1:, 0] - ball[:-1, 0]) * length,
                        (ball[1:, 1] - ball[:-1, 1]) * width) * fps
    n_seconds = -(-len(ball) // fps)
    padded = np.full(n_seconds * fps, np.nan)
    padded[:len(step)] = step
    blocks = padded.reshape(n_seconds, fps)
    out = np.full(n_seconds, np.nan)
    measured = np.isfinite(blocks).any(axis=1)
    out[measured] = np.nanmedian(blocks[measured], axis=1)
    return out


def club_names(game=GAME, data_dir=None):
    """(home, away) club names from the ATD metadata; the first team listed is home."""
    return team_names(require(["atd_meta"], game, data_dir)[0])


def _plain(name):
    """Lower case, accents removed, anything not a letter or digit turned into one space."""
    text = unicodedata.normalize("NFKD", name or "")
    text = "".join(char for char in text if not unicodedata.combining(char)).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def side_of_team_tag(tag, home, away):
    """'H', 'A' or UNASSIGNED for a Smart Tagging Team value, by similarity to the club names."""
    plain = _plain(tag)
    if not plain or plain == "n a":
        return UNASSIGNED
    home_score = difflib.SequenceMatcher(None, plain, _plain(home)).ratio()
    away_score = difflib.SequenceMatcher(None, plain, _plain(away)).ratio()
    if max(home_score, away_score) < MIN_TEAM_SIMILARITY or home_score == away_score:
        return UNASSIGNED
    return "H" if home_score > away_score else "A"


def load_tags(game=GAME, data_dir=None):
    """Smart Tagging instances as the starter loader returns them, in file order."""
    return load_smart_tagging(game, data_dir)


def tag_rows(events, home, away, fps=FPS):
    """One row per Smart Tagging instance, sorted by start, for a table of the tags.

    Frames follow the loader: start_frame inclusive, end_frame the first frame after the phase.
    """
    rows = []
    for event in sorted(events, key=lambda e: (e["start_frame"], e["end_frame"], e["code"] or "")):
        tags = event["tags"]
        rows.append({
            "Starts at (video)": video_time(event["start_frame"], fps),
            "Ends at (video)": video_time(event["end_frame"], fps),
            "Lasts (s)": round(event["end_s"] - event["start_s"], 1),
            "Code": event["code"] or "",
            "Team (as tagged)": tags.get("Team") or "",
            "Side": side_of_team_tag(tags.get("Team"), home, away),
            "Type": tags.get("Type") or "",
            "Half": tags.get("Half") or "",
            "First frame": event["start_frame"],
            "Last frame": event["end_frame"] - 1,
        })
    return rows


def tags_by_second(rows, n_seconds, fps=FPS):
    """For every second of video, the tags active during it: 'CODE (side)' joined by ' | '.

    A tag covers frames [First frame, Last frame]; it is listed under every second it touches.
    """
    active = [[] for _ in range(n_seconds)]
    for row in rows:
        first = max(0, row["First frame"] // fps)
        last = min(n_seconds - 1, row["Last frame"] // fps)
        for second in range(first, last + 1):
            active[second].append("{} ({})".format(row["Code"], row["Side"]))
    return [" | ".join(names) for names in active]
