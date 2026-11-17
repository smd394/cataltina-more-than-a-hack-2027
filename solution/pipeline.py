"""The one command that regenerates our submission.

    python solution/pipeline.py                      # -> out/submission/Cataltina.csv
    python solution/pipeline.py --out some/other.csv

Three stages:

  1. label_v0, the walking skeleton: the published baseline (nearest tracked player to the ball,
     then a centred majority vote over +/- 50 frames), rebuilt on our own arrays so the same code
     also runs on the practice matches. It must reproduce baseline_smooth50.csv frame for frame;
     rebuild_baseline.py and the tests are how we know. The rules are not written over v0
     itself but over the base (label_base), the same vote with fewer voters: "near the ball and
     not in flight". The nearest player votes for his team only when he is within 3 m of the
     ball, and nobody votes while the ball travels faster than 8 m/s, because a hard pass or a
     lob that flies past an opponent is not a change of team. Where the vote then has nobody
     for its whole window, the ball stays with the team that had it (the base's letter straight
     before that stretch; if none, straight after it; if neither, v0's), so the base has exactly
     v0's dead frames.
  2. The dead-ball rules (dead_ball.py) write D over the base, AFTER the smoothing and never
     smoothed again, so a stoppage keeps its exact first and last frame; then the lost-ball
     corrections F1 and F2.
  3. The possession rules (control.py), last: the letters the vote carried into a long ball gap
     become D (P4); then a short spell of one team that its team never touched goes back to the
     team that had the ball (P1); then, where a set-piece tag names a team and rule D found its
     stoppage, the tag's team has the ball from the restart's kick, whatever our letter says,
     until a player of the other team gets to the ball: until he touches it, or until he is the
     player near the ball, whichever comes first (P3); last, a possession shorter than 1 second
     with a dead ball on both sides becomes D (P5); last of all, a change of team is written
     where the new team first has the ball near, not in the middle of the pass that takes it
     there (control.change_on_arrival). P1, P3 and that last stage change H and A only; P4 and
     P5 write D only over an H or A. None changes a D, an X or a frame F2 wrote.

label(near_m=None, fast_m_s=None) is the program with v0 itself as the base, kept for the tests
and for measuring; the last stage and P3's near-ball end are off on that path.
label(restart_near=False) is the program with P3 as first built (until the other team's touch,
only where our letter at the kick is the other team's), and label(restart_near=False,
arrival=False) the program before the last stage. Two variants were built, measured and taken
out; the Params say what each scored and why it is not in the program.

Deterministic by construction: no randomness, no timestamps, '\\n' line endings, so two runs on the
same data give the same SHA-256 (seal 5: `python solution/check_seals.py --rebuild`).

Prints counts only.
"""

import argparse

import numpy as np

from common import (FPS, LABELS, N_FRAMES, SUBMISSION_PATH, pm, relative, run, sha256_of,
                    write_labels)
from control import change_on_arrival, control, gap_edges, islands, restart_team
from dead_ball import (SET_PIECES, SPEED_HALF_S, _runs as runs_of, dead_rules, in_flight,
                       keeper_holding, restart_stoppages)
from inputs import club_names, load_tags, side_of_team_tag
from starter.baseline import smooth
from tracking import AWAY, CHUNK, HOME, NOBODY, load_competition, nearest_team

# Params
WINDOW = 50  # half-width of the majority vote, in frames: +/- 2 s at 25 fps (baseline_smooth50)
# "Near the ball and not in flight", the base the rules are written over (label_base): the
# nearest player votes for his team only within 3 m of the tracked ball, and nobody votes while
# the tracked ball travels faster than 8 m/s. Every figure is from other matches, never ours:
# 2, 3, 5 and 8 m on Metrica's three sample games and on degraded copies of them (a noisy,
# smoothed ball; missing dots), and 2, 3 and 5 m on seven Bundesliga matches (the DFL open
# dataset: a tracked ball and the league's own possession labels); 8, 12 and 16 m/s on the
# sample games and the Bundesliga matches, as they are, with the ball lost in stoppages, and
# where touches are as scarce as on our match. Scored against v0 as the base, before the carry
# below was added (mean spellF1 gain; each set as it is and at three levels where touches are as
# scarce as on our match: one touch in four or in eight seen, or the ball smoothed over 0.8 s):
#   the sample games                                   +0.032 to +0.041 (3 of 3 games up)
#   the Bundesliga matches as they are                 +0.003 to +0.005
#   the same with the ball lost in stoppages           +0.022 to +0.034 (7 of 7 matches up)
#   ... and no dot lost, or no ball lost in play       +0.011 to +0.031
#   the practice mean (the sample games' spellF1)      0.6280 -> 0.6654
# Not below v0 in any of 60 cells: the first three sets, at the four levels, against the truth as
# it is and with one team's spells under 1, 2, 3 or 5 s between two of the other's taken out.
# Each rule alone, in this form (no D of its own): the speed test alone is not below v0 in any of
# the 60 cells either (smallest +0.0013; mean +0.0191 over the 12 cells against the truth as it
# is, the two together +0.0230). The 3 m test alone is below v0 in 5 of the 60 (on the
# Bundesliga matches as they are, -0.003 to +0.004 at the four levels); on top of the speed
# test it adds on the mean (+0.007) but is below the speed test alone in 12 of the 60. 8 m/s was
# the best of three speeds on the earlier tables; on the 3 m base 12 and 16 m/s were not above
# it in 7 and 11 of the 60 cells. How firm the result is: every neighbouring setting (2.5, 3 or
# 3.5 m with 6, 7, 8, 9 or 10 m/s) gains about the same on the mean (+0.021 to +0.024), so the
# size of the gain does not hang on the two numbers. "Not below v0 in any of 60 cells" does:
# 5 of those 15 settings pass it, and without one of the seven Bundesliga matches the built
# setting misses two cells by 0.002.
NEAR_M = 3.0
FAST_M_S = 8.0
# Where the vote has nobody in its whole window, the ball stays with the team that had it
# (label_base): the base's own letter straight before the stretch, not v0's, whose far or
# in-flight vote could otherwise come back for as little as one frame. Scored on this code
# against the same base keeping v0's letter there (mean spellF1 gain; the same sets, the four
# levels): the sample games +0.0021 to +0.0025; the Bundesliga matches as they are +0.0007 to
# +0.0027; with the ball lost in stoppages +0.0047 to +0.0056 (no dot lost: +0.0020 to +0.0039;
# no ball lost in play: +0.0035 to +0.0054); the practice mean 0.6654 -> 0.6679. Not below in
# any of the 60 cells (smallest +0.0007). Small beside the base's own gain.
# Built, measured and taken out: a base that writes D where the window holds no vote at all. It
# scored higher only on the Bundesliga matches as they are (+0.019 to +0.027), where the ball
# stands still in dead time, and lower than the base without a new D once the ball is lost in
# stoppages, as ours is; its new D stretches cut possessions in two, so a frame without a vote
# never becomes D.
# Built, measured and taken out: P3 at throw-ins, whose tags name no team, with the team of the
# dot nearest the ball at the throw (the thrower's team at 104 of 104 throw-ins on the sample
# games, 286 of 302 on the Bundesliga matches). It gained with every touch seen (+0.001 to
# +0.008) but not where touches are as scarce as on our match (-0.004 with the ball smoothed over
# 0.8 s, no match of seven up): when the other team's first touch is not seen, P3 runs on into
# their possession.
BASE = "base"                                            # the labels the rules are written over
FLIGHT, KEEPER = "ball in flight", "goalkeeper's team"   # the lost-ball corrections' stage names
GAP, CONTROL = "gap edges", "control"                    # the possession stages' names (P4, P1)
RESTART = "restart team"                                 # ... and P3's
BEFORE = "before possession"                             # the labels the possession stages receive
RESTARTS = "tagged restarts"                             # P3's [(kick frame, "H" / "A")]
ISLANDS = "dead islands"                                 # P5's stage name
ARRIVAL = "change on arrival"                            # the last stage's name
# The last stage (label(arrival=True), control.change_on_arrival): a change of team is written
# where the new team first has the ball near. The base's vote changes team in the middle of a
# pass (half the voters on each side); the possession changes when the new team gets the ball.
# It has no number of its own: the voter is the base's (a dot within NEAR_M, the ball not faster
# than FAST_M_S) and the search reaches WINDOW frames; the change moves by at most half the new
# possession and writes no D. It adds or removes no possession, except where a frame the
# goalkeeper rule (F2) wrote lies inside the moved stretch: that frame keeps its letter and can
# split the possession (once in the three sample games; never on our match). How it was found,
# said plainly: it came out of a check of the measuring setup AFTER earlier tables had been read
# (every outside set scored higher with our changes of team moved later), and its form was set
# after a first table of fixed delays; it was then scored once, as built, against a gate fixed
# beforehand, on other matches only (mean spellF1 gain against the program without it; the
# sample games and the Bundesliga matches with the ball lost in stoppages, each as it is and at
# the three levels where touches are as scarce as on our match):
#   the sample games                                    +0.0212 to +0.0425 (3 of 3 games up)
#   the Bundesliga matches, ball lost in stoppages      +0.0229 to +0.0656 (7 of 7 matches up)
#   two more such benches (the mean of the four levels) +0.0428 and +0.0403
#   the Bundesliga matches as they are (not in the gate) +0.0297 to +0.0598
#   the practice mean (the sample games' spellF1)       0.6679 -> 0.7105
# Not below the program without it in any of 40 cells (the first two sets, the four levels, the
# truth as it is and with one team's spells under 1, 2, 3 or 5 s between two of the other's taken
# out; smallest +0.0099); the worst single match in those cells +0.0139. Each of its three
# borrowed numbers moved alone (2 or 5 m, 12 m/s or no speed test, a 100-frame search) still
# passes every line; a 25-frame search does not. On the outside sets, the changes of team that
# can be paired with the truth's within 1 s rise by a quarter to a half, and the share of them
# within 8 frames of it from 35-42% to 60-79%.
# P3's second end (label(restart_near=True): control.restart_team handed the base's voter): "A
# corner, a goal kick or a free kick is taken by one team. From the kick that team has the ball,
# whatever our letter says, until a player of the other team gets to the ball: until he touches
# it, or until he is the player near the ball, whichever comes first." It has no number of its
# own: "near" is the base's voter (a dot within NEAR_M, the ball not faster than FAST_M_S), the
# taker and the 0.4 s after the kick are P3's (control.taker, control.KICK_GUARD_S). How it was
# found, said plainly: sixteen forms of P3 (four ends, acting only where our letter at the kick
# is the other team's or from every kick, with and without the Laws' zone around the ball) were
# written down before any was scored; this was the one that passed a gate fixed beforehand, and
# it was scored again as built. On other matches only, against P3 without it with the same
# restarts (made from the matches' own events: corners, free kicks and goal kicks); mean spellF1
# gain, each set as it is and at the three levels where touches are as scarce as on our match:
#   the sample games                                     +0.0013 to +0.0058 (1 to 2 of 3 up)
#   the Bundesliga matches, ball lost in stoppages       +0.0009 to +0.0140 (4 to 6 of 7 up)
#   two more such benches (the mean of the four levels)  +0.0061 and +0.0078
#   the Bundesliga matches as they are (not in the gate) +0.0005 to +0.0144
#   the sample games with those restarts                 0.7154 -> 0.7167
# Not below P3 without it in any of 40 cells (the first two sets, the four levels, the truth as it
# is and with one team's spells under 1, 2, 3 or 5 s between two of the other's taken out;
# smallest +0.0008); the worst single match in those cells -0.0023. With the restarts found by the
# program itself from tags made from those events (rule D's restart as the kick; all tags, half
# or four in five of them dropped, or no free kicks), above zero in 32 of 32 rows (smallest
# +0.0002). On a second draw of the benches' degradations one of the 40 cells read -0.0001. The
# gain is largest where touches are scarce: P3 no longer runs on into the other team's
# possession when their first touch is not seen. What no outside set can show: a dot of the
# other team's colour that is in fact a teammate ends it too, and only our tracking has such
# dots. The practice mean cannot move: the practice games have no Smart Tagging, so P3 does not
# run there.


def label_v0(tracking, window=WINDOW):
    """H / A / D per frame. v0: nearest player's team, smoothed; no ball for +/- window -> D."""
    return smooth(nearest_team(tracking), window)


def ball_speed(tracking, fps=FPS):
    """Per frame, the tracked ball's speed in m/s: its displacement over SPEED_HALF_S either side
    of the frame (the way dead_ball.player_speeds measures a player), never across a period's
    edge; NaN where the ball is not tracked at both ends, and outside the periods."""
    h = max(1, int(round(SPEED_HALF_S * fps)))
    xy = tracking.ball * np.array(tracking.pitch)
    speed = np.full(tracking.n_frames, np.nan)
    for _, first, stop in tracking.blocks or [(None, 0, tracking.n_frames)]:
        if stop - first > 2 * h:
            step = xy[first + 2 * h:stop] - xy[first:stop - 2 * h]
            speed[first + h:stop - h] = np.hypot(step[:, 0], step[:, 1]) * fps / (2.0 * h)
    return speed


def near_team(tracking, near_m, fast_m_s=None):
    """Per frame, HOME / AWAY for the team of the tracked player nearest the ball when he is
    within near_m metres of it, else NOBODY: tracking.nearest_team's answer, taken away on the
    frames where even the nearest tracked dot is further than near_m from the tracked ball
    (near_m None: no distance test). With fast_m_s, also taken away on the frames where the
    tracked ball travels faster than fast_m_s (ball_speed); where its speed cannot be measured
    the dot votes. No dot is moved or dropped; a far dot, or any dot while the ball flies past,
    only does not vote."""
    out = nearest_team(tracking).copy()
    length, width = tracking.pitch
    starts = range(0, tracking.n_frames, CHUNK) if near_m is not None else ()
    for start in starts:
        ball = tracking.ball[start:start + CHUNK]
        players = tracking.players[start:start + CHUNK]
        if players.shape[1] == 0:
            continue
        distance = np.hypot(players[:, :, 0] * length - (ball[:, 0] * length)[:, None],
                            players[:, :, 1] * width - (ball[:, 1] * width)[:, None])
        nearest = np.where(np.isnan(distance), np.inf, distance).min(axis=1)
        out[start:start + CHUNK][nearest > near_m] = NOBODY
    if fast_m_s is not None:
        with np.errstate(invalid="ignore"):             # an unknown speed (NaN) is not fast
            out[ball_speed(tracking) > fast_m_s] = NOBODY
    return out


def label_base(tracking, window=WINDOW, near_m=NEAR_M, fast_m_s=FAST_M_S, v0=None):
    """H / A / D per frame: the base the rules are written over, in place of v0 (label() uses it
    unless near_m and fast_m_s are both None).

    "The nearest player votes for his team only when he is within 3 m of the ball. A far dot
    says little about who has the ball: the ball is travelling, or the player who has it has no
    dot."

    With fast_m_s: "A hard pass or a lob that flies past an opponent is not a change of team. So
    a player does not vote for his team while the ball is travelling faster than 8 m/s; the
    letter changes when the ball slows down at the other team."

    "Where nobody is near the ball, or the ball is in flight, for the whole of the vote's
    window, nobody votes. There the ball stays with the team that had it: possession stays with
    the last team that had it while the ball is loose."

    The same vote as v0 (the majority over +/- window frames), taken over near_team: a frame
    whose nearest tracked dot is further than near_m from the ball, or whose ball is faster than
    fast_m_s, gives no vote. Where the whole window holds no vote, the vote would say D. In each
    such stretch of frames that v0 gives to a team, every frame takes the base's own letter
    straight before the stretch; if there is none (the frame before is D, or the stretch starts
    the file), the base's letter straight after it; if neither, v0's letter. Only frames v0
    gives to a team are filled, so the base has exactly v0's D frames and writes no D of its
    own. v0 = label_v0(tracking, window), passed in by a caller that already has it.
    """
    base = smooth(near_team(tracking, near_m, fast_m_s), window)
    v0 = label_v0(tracking, window) if v0 is None else v0
    dead = LABELS[pm.D]
    loose = (base == dead) & (v0 != dead)
    for first, stop in runs_of(loose):
        before = base[first - 1] if first > 0 and base[first - 1] != dead else None
        after = base[stop] if stop < len(base) and base[stop] != dead else None
        if before is not None:
            base[first:stop] = before
        elif after is not None:
            base[first:stop] = after
        else:
            base[first:stop] = v0[first:stop]
    return base


def tagged_restarts(tracking, tags, names):
    """P3's restarts: [(kick frame, "H" / "A")] for every SET PIECES tag that names a team (by
    inputs.side_of_team_tag against names = (home, away)) and whose restart rule D found
    (dead_ball.restart_stoppages: the kick is the stoppage's restart frame). Empty without tags
    or names, as on the practice games.
    """
    if tags is None or names is None:
        return []
    teams = (LABELS[pm.H], LABELS[pm.A])
    out = []
    for tag, stretch in restart_stoppages(tracking, tags):
        if stretch is None or (tag["code"] or "").strip().lower() != SET_PIECES:
            continue
        side = side_of_team_tag(tag["tags"].get("Team"), *names)
        if side in teams:
            out.append((stretch[1], side))
    return out


def label(tracking, tags=None, window=WINDOW, stages=None, names=None, near_m=NEAR_M,
          fast_m_s=FAST_M_S, arrival=True, restart_near=True):
    """H / A / D / X per frame: the base, label_base(near_m, fast_m_s), in place of v0
    (near_m and fast_m_s both None: v0 itself, the program before the base), then every
    dead-ball rule written over it as D (dead_rules:
    B, B2, C, F3, and with tags D, D2, D'), then the lost-ball corrections: a ball in flight is X
    (F1), a ball the goalkeeper holds is his team's (F2). Neither writes over any dead-ball rule.
    Last, the possession rules: P4 (control.gap_edges), P1 (control.control), then with tags and
    names P3 (control.restart_team; with restart_near it is handed the base's voter
    near_team(near_m, fast_m_s), so a restart team keeps the ball from the kick until the other
    team touches the ball or is the one near it, whichever comes first; without, P3 as first
    built: until the other team's touch, only where our letter at the kick is the other
    team's), then P5 (control.islands), then, with arrival, every change of team moved to where
    the new team first has the ball near (control.change_on_arrival, over the same voter).
    restart_near and arrival are both off whenever near_m and fast_m_s are both None. None
    changes a D, an X or a frame F2 wrote.

    The overlay comes after the smoothing. tags are the Smart Tagging instances
    (inputs.load_tags()), read by the restart rules D and D'; None runs only the rules that need
    tracking alone, as on the practice games, which have no Smart Tagging (P4 and P1 run on both).
    names = (home, away) club names (inputs.club_names()), read by P3 to know which team a tag
    names; without them P3 does nothing. stages, a dict if given, receives "v0" (always the raw
    smoothed baseline), the base the rules are written over (BASE), each dead rule's mask by
    name, the frames F1 and F2 wrote, the labels the possession rules receive (BEFORE), the
    frames P4 wrote D (GAP), P1 rewrote (CONTROL), P3 rewrote (RESTART), P5 wrote D
    (ISLANDS) and the last stage rewrote (ARRIVAL; none when it is off), and P3's restarts
    (RESTARTS). tracking is left unchanged.
    """
    v0 = label_v0(tracking, window)
    stages = {} if stages is None else stages
    stages["v0"] = v0.copy()
    labels = v0 if near_m is None and fast_m_s is None else \
        label_base(tracking, window, near_m=near_m, fast_m_s=fast_m_s, v0=v0)
    stages[BASE] = labels.copy()
    dead = np.zeros(len(labels), dtype=bool)
    for name, mask in dead_rules(tracking, tags):
        stages[name] = mask
        dead |= mask
    labels[dead] = LABELS[pm.D]
    free = ~dead
    stages[FLIGHT] = in_flight(tracking) & free & (labels == LABELS[pm.D])
    labels[stages[FLIGHT]] = LABELS[pm.X]
    held = keeper_holding(tracking)
    stages[KEEPER] = (held != NOBODY) & free
    for team, code in ((HOME, pm.H), (AWAY, pm.A)):
        labels[(held == team) & free] = LABELS[code]
    stages[BEFORE] = labels.copy()
    labels, stages[GAP] = gap_edges(labels, tracking, stages[KEEPER], window)
    labels, stages[CONTROL] = control(labels, tracking, stages[KEEPER])
    votes = None
    if (arrival or restart_near) and not (near_m is None and fast_m_s is None):
        votes = near_team(tracking, near_m, fast_m_s)        # the base's own voter
    stages[RESTARTS] = tagged_restarts(tracking, tags, names)
    labels, stages[RESTART] = restart_team(labels, tracking, stages[RESTARTS], stages[KEEPER],
                                           votes=votes if restart_near else None)
    labels, stages[ISLANDS] = islands(labels, tracking, stages[KEEPER])
    stages[ARRIVAL] = np.zeros(len(labels), dtype=bool)
    if arrival and votes is not None:
        labels, stages[ARRIVAL] = change_on_arrival(labels, tracking, votes, stages[KEEPER],
                                                    window)
    return labels


def summary(labels, blocks):
    """(label counts, possessions, mean possession seconds) inside the periods of play."""
    codes = pm.label_codes(labels)
    runs = [(first, stop) for _, first, stop in blocks]
    stats = pm.possession_stats(pm.fill_abstentions(codes, runs), runs)
    counts = {name: int(np.sum(codes == k)) for k, name in enumerate(LABELS)}
    return counts, stats["n_spells"], stats["mean_duration_s"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=SUBMISSION_PATH, help="where to write the CSV")
    parser.add_argument("--data-dir", help="the data bundle, if not in data/")
    args = parser.parse_args()

    tracking = load_competition(data_dir=args.data_dir)
    tags = load_tags(data_dir=args.data_dir)
    stages = {}
    labels = label(tracking, tags, stages=stages, names=club_names(data_dir=args.data_dir))
    path = write_labels(args.out, labels)

    reason = pm.validate_submission(pm.read_label_csv(path), N_FRAMES)
    counts, n_possessions, mean_s = summary(labels, tracking.blocks)
    print("wrote {}".format(relative(path)))
    print("  valid: {}".format("yes" if reason is None else "NO -- " + reason))
    print("  labels  " + "  ".join("{} {}".format(name, counts[name]) for name in LABELS))
    print("  {} possessions in the graded halves, mean {:.1f} s".format(n_possessions, mean_s))
    graded = np.zeros(len(labels), dtype=bool)
    for _, first, stop in tracking.blocks:
        graded[first:stop] = True
    dead = labels == LABELS[pm.D]

    def final(name, where=graded):
        """Frames of a stage that are D in the file as written (a rule can overlap another)."""
        return int(np.sum(stages[name] & where & dead))

    print("  dead-ball rules, D in the file, graded frames: before kickoff {}, ball off the "
          "pitch {}, restarts from the tags {}".format(
              final("before kickoff"), final("ball off the pitch"),
              final("restarts from the tags")))
    print("  every stage together (the base, the dead-ball rules, F1, F2, P4, P1, P3, P5, the "
          "change on arrival), "
          "graded frames changed from v0: {}".format(
              int(np.sum((labels != stages["v0"]) & graded))))
    print("  the base (near the ball, not in flight), graded frames changed from v0: {}".format(
        int(np.sum((stages[BASE] != stages["v0"]) & graded))))
    print("  lost-ball rules, graded frames: huddle D {}, ball in flight X {}, "
          "goalkeeper's team {}".format(
              final("huddle"), int(np.sum(stages[FLIGHT] & graded)),
              int(np.sum(stages[KEEPER] & graded))))
    print("  outside the halves (never graded), frames written D: {}".format(
        final("outside the halves", np.ones(len(labels), dtype=bool))))
    print("  possession rules, graded frames: P4 gap edges written D {}, P1 rewritten {}; "
          "possessions {} before them".format(
              int(np.sum(stages[GAP] & graded)), int(np.sum(stages[CONTROL] & graded)),
              summary(stages[BEFORE], tracking.blocks)[1]))
    # a restart's rewritten frames lie between its kick and the next restart's kick (each kick
    # ends a stoppage, and P3 stops at the first dead ball after its own kick)
    kicks = sorted({kick for kick, _ in stages[RESTARTS]})
    acted = sum(bool(stages[RESTART][first:stop].any())
                for first, stop in zip(kicks, kicks[1:] + [len(labels)]))
    print("  possession rule P3, tagged restarts naming a team: {}; that rewrote letters {}; "
          "graded frames rewritten {}".format(len(stages[RESTARTS]), acted,
                                              int(np.sum(stages[RESTART] & graded))))
    island = (stages[ISLANDS] & graded).astype(np.int8)
    print("  possession rule P5, possessions under 1 s between two dead balls written D: "
          "stretches {}, graded frames {}".format(
              int(np.sum(np.diff(np.concatenate(([0], island))) == 1)), int(np.sum(island))))
    moved = (stages[ARRIVAL] & graded).astype(np.int8)
    print("  changes of team moved to where the new team first has the ball near: stretches {}, "
          "graded letters moved {}".format(
              int(np.sum(np.diff(np.concatenate(([0], moved))) == 1)), int(np.sum(moved))))
    print("  SHA-256 {}".format(sha256_of(path)))
    if reason is not None:
        raise SystemExit(1)


if __name__ == '__main__':
    run(main)
