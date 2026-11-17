"""Tracking as arrays, whichever match it comes from, and the nearest-player rule on them.

The competition ATD and the practice matches arrive in different formats. Each loader turns its
format into one Tracking object, so every labelling rule is written once and runs on both:

    ball      (n, 2)     normalised pitch x, y per video frame; NaN where not tracked
    players   (n, s, 2)  normalised x, y per track slot; NaN where the slot is empty
    team      (s,)       0 = home, 1 = away, per slot
    pitch     (length, width) in metres, to measure distances in metres
    blocks    [(half, first_frame, stop_frame)] of the periods of play, half-open
    kickoffs  {half: video frame of the kickoff}, or None where unknown (the practice games)

Index i is video frame i, counting from 0, exactly as in a submission.
"""

from dataclasses import dataclass, field

import numpy as np

from common import GAME, GAMES, N_FRAMES, graded_blocks, kickoff_frames
from games import require
from starter.baseline import AWAY, HOME, NOBODY, team_of_each_track

# Params
# Raw ATD line: "<frame_id>:<x,y,speed>;<x,y,speed>;...:<x,y,z>", "NaN" where not observed.
FIELD_SEP, GROUP_SEP, VALUE_SEP = ":", ";", ","
ATD_VALUES_PER_PLAYER = 3
CHUNK = 20000  # frames per vectorised step: bounds memory to a few tens of MB


@dataclass
class Tracking:
    ball: np.ndarray
    players: np.ndarray
    team: np.ndarray
    pitch: tuple
    blocks: list = field(default_factory=list)
    kickoffs: dict = None

    @property
    def n_frames(self):
        return len(self.ball)


def read_atd(raw_path, team, n_frames=N_FRAMES, frames=None):
    """Parse a raw ATD file into (ball, players) arrays indexed by video frame.

    ATD frame_id k is video frame k - 1. `frames` = (first, stop) keeps only that window, so a
    caller that needs twenty seconds of video does not parse two hours of tracking.
    """
    n_slots = len(team)
    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, n_slots, 2), np.nan)
    first, stop = frames if frames else (0, n_frames)
    with open(raw_path) as handle:
        for line in handle:
            head = line[:line.find(FIELD_SEP)]
            if not head.isdigit() or not first <= int(head) - 1 < stop:
                continue
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(head) - 1
            values = np.array(parts[1].replace(GROUP_SEP, VALUE_SEP).split(VALUE_SEP),
                              dtype=np.float64).reshape(-1, ATD_VALUES_PER_PLAYER)[:n_slots, :2]
            players[frame, :len(values)] = values
            ball[frame] = np.array(parts[2].split(VALUE_SEP)[:2], dtype=np.float64)
    return ball, players


def load_competition(game=GAME, data_dir=None, frames=None):
    """The tactical ATD of the competition match as a Tracking."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_each_track(meta_path)
    ball, players = read_atd(raw_path, team, GAMES[game]["n_video_frames"], frames)
    return Tracking(ball, players, team, GAMES[game]["pitch"], graded_blocks(game),
                    kickoff_frames())


def nearest_team(tracking):
    """Per frame, HOME / AWAY for the team of the tracked player nearest the ball, else NOBODY.

    The published baseline's rule, vectorised: distances in metres (both axes scaled, since the
    pitch is not square), the first slot wins an exact tie, and a frame without a tracked ball or
    without any tracked player has no answer.
    """
    length, width = tracking.pitch
    out = np.full(tracking.n_frames, NOBODY, dtype=np.int8)
    for start in range(0, tracking.n_frames, CHUNK):
        ball = tracking.ball[start:start + CHUNK]
        players = tracking.players[start:start + CHUNK]
        dx = players[:, :, 0] * length - (ball[:, 0] * length)[:, None]
        dy = players[:, :, 1] * width - (ball[:, 1] * width)[:, None]
        distance = dx * dx + dy * dy
        distance = np.where(np.isnan(distance), np.inf, distance)
        nearest = distance.argmin(axis=1)
        found = np.isfinite(distance[np.arange(len(nearest)), nearest])
        out[start:start + CHUNK][found] = tracking.team[nearest[found]]
    return out


__all__ = ["Tracking", "read_atd", "load_competition", "nearest_team", "HOME", "AWAY", "NOBODY"]
