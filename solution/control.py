"""Who has the ball: the possession rules, written over the labels after the dead-ball rules.

Our labels cut play into too many possessions. The nearest player to the ball changes team every
time an opponent comes close, even when he never gets the ball, and the scorer matches possessions
one to one: a short false spell in the middle of a real possession leaves three spells where the
truth has one, so it costs the real possession AND adds a false one. A merge that swallows a real
short possession does the same damage the other way, so a spell is kept whenever its team touched
the ball.

What the challenge says (CHALLENGE.md):
  55-56  "A change of possession occurs when a player from the opposing team gains **controlled**
         possession of the ball."
  63-64  "In phases where the ball is contested or loose, possession stays with the last team that
         had it."
  78-80  not a change: "a simple touch of the ball without control", "a deflection of a pass that
         does not intercept it", "an aerial duel won without subsequent control".
The team's first reading: "a contested ball stays with the team that had it." Then, after
watching the video beside the organisers' public correct answers: "the official answer switches
at the new team's first touch when that touch plays the ball", never while the ball is still
travelling to them; "it follows more the change of direction rather than touch, at least in messy
cases; the last touch feels more like a confirmation."

The rule, in plain words (the wording the team reads):

  P1 A short spell is not a possession unless its team touched the ball. A spell of one team
     shorter than 5 seconds, with the other team's spell straight before it and either the other
     team's spell or a dead ball straight after it, becomes the other team's: the ball never really
     changed hands. It is kept when its team touched the ball during it: the ball changed speed or
     direction sharply (by a kick's worth, 4 m/s or more) while one of that team's players was
     within 1.5 m of it. The change is the sign; the player at the ball confirms it, and he need
     not be the nearest one (in a crowd, which dot is nearest is chance). The shortest spell is
     looked at first, the earliest on a tie, and the spells are read again after each change, so
     the result does not depend on where reading starts.

  P4 "The vote keeps a team's letter for 2 seconds after the tracker loses the ball, and gives
     one 2 seconds before it finds the ball again. In a long gap those letters are a guess; on
     the sample games four in five of them are dead ball. They become D." Exactly: in a gap the
     tracker loses the ball for more than 4 seconds (longer than the vote can bridge) and that
     holds a dead ball somewhere, every H or A frame becomes D, wherever it sits in the gap,
     except the frames the goalkeeper rule (F2) wrote. Where rule D2 starts a stoppage later
     inside such a gap (the players still running), P4 still writes D before that start.

  P3 "A corner, a goal kick or a free kick is taken by one team. From the kick that team has the
     ball, whatever our letter says, until a player of the other team gets to the ball: until
     he touches it, or until he is the player near the ball, whichever comes first. The player
     nearest the ball at the kick is the taker, whatever colour his dot shows." Exactly: only at
     a tag that names a team and for which rule D wrote a stoppage (it found the ball's loss and
     the kick), and only where our letter at the kick is a team's (a kick on a dead ball or an X
     changes nothing); from the kick up to the first dead ball or frame F2 wrote, every letter
     of the other team becomes the restart team's, until the earlier of the other team's first
     touch and the first frame where the base's voter (the nearest dot within 3 m of the ball,
     the ball not faster than 8 m/s) is a dot of the other team that is not the taker. For the
     first 0.4 seconds after the kick neither is counted (the touch test still sees the kick
     itself; the vote keeps the same guard). (As first built, without the voter, P3
     acted only where our letter at the kick was the other team's and ended at their touch:
     where touches are as scarce as on our match, a first touch not seen let it run on into the
     other team's possession.)

  P5 "A possession shorter than 1 second with a dead ball on both sides is not a possession:
     nobody restarts play and loses the ball again that fast. It is dead." (Rule C's reasoning:
     two out stretches less than 1 second apart are one stoppage; the same 1 second.) The start
     or end of a half counts as a dead ball; a spell holding a frame F2 wrote is kept.

  Last, the change on arrival: "A change of team is written where the new team first has the
     ball near. The vote changes team in the middle of a pass; the possession changes when the
     new team gets the ball." Exactly: where one team's spell follows the other team's straight
     away, the change moves to the first frame, within the vote's window, where the base's
     voter (a dot of the new team within 3 m of the ball, the ball not faster than 8 m/s) names
     the new team; never by more than half the new spell. With no such frame nothing moves. A
     spell that follows a dead ball is a restart, not a pass, and is not touched. It moves a
     change later, never earlier, and writes no D; it adds or removes no possession, except
     where a frame F2 wrote sits inside the moved stretch (that frame keeps its letter).

  P1, P3 and the change on arrival never change a dead ball (D) or a frame where we abstain (X);
  P4 and P5 write D only over an H or A. No rule joins spells across a dead ball or across
  halftime; P1 and P5 never change a spell that holds a frame the goalkeeper rule (F2) wrote,
  and P3, P4 and the change on arrival never change such a frame. Spells are read the way the
  scorer reads them: an X repeats the label before it.

Measured on the sample games, never on our match, when the rule was first built:
  - A spell with a dead ball straight BEFORE it and the other team after it is not merged: that
    variant lowered the practice score on two of the three games.
  - "Control time" (frames where the team's player is the nearest dot within reach) gave no
    exception: the short spells the rule merges show none, so the rule has no control-time clause.

Nothing here reads files or prints.
"""

import numpy as np

from common import FPS, LABELS, pm
from dead_ball import MIN_OUT_S, SPEED_HALF_S
from tracking import AWAY, HOME

# Params
# Each number is measured on Metrica's three sample games with their practice answers, never on
# our match: numbers are measured and tuned on outside data only.
# The reach and the change were chosen by a cautious rule (2026-11-14), BEFORE P4 existed: over
# the grid reach 0.4, 0.7, 1.0, 1.5, 2.0 m x change 0.5, 1, 2, 3, 4 m/s x short 2, 3, 4, 5, 6,
# 8 s, among the settings with no game below its score without P1 and the
# touch-style mean not below its own (all 150 qualify), BEST = 0.6237 (0.4 m, 3 m/s, 8 s); the 30
# settings within 0.01 of it are cautious; of those, the largest change, then the widest reach,
# then the shortest spell: 1.5 m, 4 m/s, 8 s, mean 0.6139 (0.6235 / 0.6354 / 0.5827). Leaving one
# game out picks (1.0, 4, 8), (1.5, 4, 8), (1.5, 4, 5). With P4 in place the same rule would pick
# 1.0 m (an independent review of the code, 2026-11-14); the wider reach is kept as the careful
# side on a tracking whose dots sit further from the ball. (The first rules gave 0.4 m and
# 0.5 m/s: they measured a sample ball that sits on the player and travels in straight lines.)
# "Short": a spell under 5 s. How the length was set (2026-11-14, the three sample games with
# our practice answers, P4 in place): the longest length at which P1 with a touch test that sees
# nothing (touch_m_s infinite) is still not below having no P1 at all. Mean spellF1 over the
# three games:
#   short          2 s     3 s     4 s     5 s     6 s     8 s      no P1 at all 0.5350
#   touch as built 0.6079  0.6204  0.6241  0.6265  0.6292  0.6325
#   never seen     0.5993  0.5885  0.5641  0.5387  0.5054  0.4714
# At 5 s as built no game is more than 0.01 below its score at 8 s (-0.0016 / -0.0080 / -0.0083);
# leaving one game out picks 5, 5, 4 s, and each left-out game there is above its score without
# P1.
# What that is worth (measured again on real labels, 2026-11-14 to 16): "at 5 s a touch test
# that sees nothing is never worse than no rule" holds on OUR PRACTICE ANSWERS ONLY. On seven
# Bundesliga matches with the league's own possession labels a blind P1 at 5 s is BELOW no P1
# (0.3413 against 0.4022; not below it only up to 2 s), and on the same matches with the ball
# lost in stoppages too (0.3787 against 0.3920; not below up to 4 s). And the length itself turns
# on what the truth holds. With touches as scarce as on our match (one in four or one in eight
# seen, or the ball smoothed over 0.8 s), 4 s against 5 s scores HIGHER against truths that
# hold many short possessions (+0.004 to +0.026 in 36 of 36 cells: the truths as they are, or
# with one team's spells under 1, 2 or 3 s between two of the other team's taken out) and
# LOWER against truths that hold few (-0.003 to -0.019 in 9 of 9: those under 5 s taken out);
# with every touch seen 4 s is lower everywhere (-0.0004 to -0.005, 15 of 15). How many short
# possessions the truth of our match holds cannot be measured, so the length stays at 5 s.
SHORT_S = 5.0
# "At the ball": a player of the team within 1.5 m of the tracked ball (our dots sit further from
# the ball than the sample games' do).
REACH_M = 1.5
# "The ball changes speed or direction": the size of the change of its velocity, over the 0.2 s
# after a frame against the 0.2 s before (dead_ball.SPEED_HALF_S), at least 4 m/s: a kick's worth
# (dead_ball.MOVING_M_S is the grid's physical end), not tracking jitter.
TOUCH_M_S = 4.0
# A short spell with the other team before it and a dead ball after it (a touch that puts the ball
# out) is merged too: +0.023 on the practice mean, no game falling, under both sets of practice
# answers. One with a dead ball before it and the other team after it is not (-0.002, two games
# fall).
MERGE_BEFORE_DEAD = True
# P4 has no number of its own (a long gap is one the vote cannot bridge, 2 x pipeline.WINDOW).
# Measured on the sample games, P1 at the numbers above (floor 0.6139): the gap's start only
# 0.6261, its end only 0.6195, both 0.6322. Then (2026-11-14) every carried letter of the gap, not
# only those before its first D and after its last: 0.6325 (0.6370 / 0.6578 / 0.6027; 8 more
# frames D, all on game3; touch-style 0.6317 -> 0.6320).
# P3: the taker is the tracked dot nearest the ball at the kick, if within 9.15 m (Law 13: the
# opponents stand at least 9.15 m from the ball at a free kick and a corner, Law 16 at a goal
# kick).
TAKER_M = 9.15
# P3: no touch of the other team is counted in the first 0.4 s after the kick. The touch test
# measures the ball's change over SPEED_HALF_S each side of a frame, so for that long after the
# kick its window still holds the kick itself; derived from the test's own window, not measured.
# On the sample games, restarts made from their events: 0.6394 with it and without it (it
# changes no frame there; the sample ball sits on the taker, so no second dot is at the kick).
KICK_GUARD_S = 2 * SPEED_HALF_S
# P5 has no number of its own: "under 1 second" is dead_ball.MIN_OUT_S, rule C's. Measured
# before it was wired, on the sample games, everything above in place: 0.6265 -> 0.6280
# (0.6354 / 0.6532 / 0.5953, no game falling; touch-style 0.6261 -> 0.6275); 4 islands, 1 of
# them real in the answers.
# The change on arrival has no number of its own either: its voter, its reach and its window are
# the base's (pipeline.near_team, NEAR_M, FAST_M_S, WINDOW), passed in by pipeline.label. What it
# scored on other matches, and how it was found, is in pipeline.py's Params. The same holds for
# P3's second end, the other team near the ball (restart_team's votes, the base's voter): no
# number of its own; its scores are in pipeline.py's Params.


def _blocks(tracking):
    """The periods of play; a Tracking without blocks is one period over all its frames."""
    return tracking.blocks or [(None, 0, tracking.n_frames)]


def ball_change(tracking, fps=FPS):
    """Per frame, the size in m/s of the change of the ball's velocity: over the SPEED_HALF_S after
    the frame against the SPEED_HALF_S before it. NaN unless the ball is tracked on the frame and
    at both ends; never measured across a period's edge."""
    h = max(1, int(round(SPEED_HALF_S * fps)))
    xy = tracking.ball * np.array(tracking.pitch)
    out = np.full(tracking.n_frames, np.nan)
    for _, first, stop in _blocks(tracking):
        if stop - first <= 2 * h:
            continue
        f = np.arange(first + h, stop - h)
        change = (xy[f + h] - xy[f]) * fps / h - (xy[f] - xy[f - h]) * fps / h
        out[f] = np.hypot(change[:, 0], change[:, 1])
    return out


def within_reach(tracking, reach_m=None, chunk=20000):
    """(n, 2) True where some tracked dot of home (column HOME) / away (AWAY) is within reach_m
    (default REACH_M, read when called) of the tracked ball."""
    reach_m = REACH_M if reach_m is None else reach_m
    length, width = tracking.pitch
    out = np.zeros((tracking.n_frames, 2), dtype=bool)
    for start in range(0, tracking.n_frames, chunk):
        ball = tracking.ball[start:start + chunk]
        players = tracking.players[start:start + chunk]
        dx = players[:, :, 0] * length - (ball[:, 0] * length)[:, None]
        dy = players[:, :, 1] * width - (ball[:, 1] * width)[:, None]
        with np.errstate(invalid="ignore"):
            near = np.hypot(dx, dy) <= reach_m          # NaN (untracked) compares False
        for team in (HOME, AWAY):
            out[start:start + chunk, team] = near[:, tracking.team == team].any(axis=1)
    return out


def touches(tracking, fps=FPS, reach_m=None, touch_m_s=None):
    """{H: mask, A: mask} of the frames where that team touches the ball: the ball's velocity
    changes by touch_m_s or more while a player of the team is within reach_m of it (defaults
    TOUCH_M_S and REACH_M, read when called)."""
    touch_m_s = TOUCH_M_S if touch_m_s is None else touch_m_s
    changed = np.nan_to_num(ball_change(tracking, fps), nan=-1.0) >= touch_m_s
    near = within_reach(tracking, reach_m)
    return {pm.H: changed & near[:, HOME], pm.A: changed & near[:, AWAY]}


def _counter(mask):
    """count(a, b) = the number of True frames of mask in [a, b), in constant time."""
    total = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
    return lambda a, b: int(total[b] - total[a])


def _candidates(segments):
    """(index, start, stop, code) of each spell with P1's neighbours: the other team straight
    before it, and the other team (or, with MERGE_BEFORE_DEAD, a D) straight after it."""
    for i, (a, b, k) in enumerate(segments):
        if k not in pm.POSSESSION_LABELS or i == 0:
            continue
        other = 1 - k                                       # H = 0, A = 1 in the scorer's codes
        after = segments[i + 1][2] if i + 1 < len(segments) else None
        if segments[i - 1][2] == other and (
                after == other or (MERGE_BEFORE_DEAD and after == pm.D)):
            yield i, a, b, k


def control(labels, tracking, protected, fps=FPS, short_s=None, reach_m=None, touch_m_s=None):
    """P1 over a label array: (new labels, mask of the frames rewritten). Pure: labels, tracking
    and protected (the frames F2 wrote, a boolean mask) are left unchanged.

    Inside each period, the spells are read the scorer's way (X filled with the label before it,
    pm.fill_abstentions, then pm.segments). Among the spells with P1's neighbours (_candidates)
    that are shorter than short_s and hold no touch by their team and no protected frame, the
    shortest (the earliest on a tie) becomes the other team's; the spells are read again, until
    none is left. Only H and A frames are rewritten; D and X frames never are. short_s, reach_m
    and touch_m_s default to SHORT_S, REACH_M and TOUCH_M_S, read when called (so the numbers can
    be measured on other games).
    """
    codes = pm.label_codes(labels)
    out = codes.copy()
    short = int(round((SHORT_S if short_s is None else short_s) * fps))
    touched = {k: _counter(mask) for k, mask in
               touches(tracking, fps, reach_m, touch_m_s).items()}
    guarded = _counter(np.asarray(protected, dtype=bool))
    for _, first, stop in _blocks(tracking):
        run = [(first, stop)]
        segments = [list(s) for s in pm.segments(pm.fill_abstentions(codes, run), run)]
        while True:
            best = None
            for i, a, b, k in _candidates(segments):
                if b - a >= short or touched[k](a, b) or guarded(a, b):
                    continue
                if best is None or (b - a, a) < (segments[best][1] - segments[best][0],
                                                 segments[best][0]):
                    best = i
            if best is None:
                break
            a, b, k = segments[best]
            part = out[a:b]
            part[part == k] = 1 - k
            segments[best][2] = 1 - k
            if best + 1 < len(segments) and segments[best + 1][2] == 1 - k:
                segments[best][1] = segments[best + 1][1]
                del segments[best + 1]
            if segments[best - 1][2] == 1 - k:
                segments[best - 1][1] = segments[best][1]
                del segments[best]
    changed = out != codes
    return np.array(LABELS)[out].astype(np.asarray(labels).dtype), changed


def _runs(mask):
    """[(start, stop)] of the True runs of a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def gap_edges(labels, tracking, protected, window):
    """P4 over a label array: (new labels, mask of the frames written D). Pure, like control().

    Inside each period, for every run of frames with no tracked ball longer than 2 x window
    frames (longer than the vote can bridge) that holds at least one D or X frame, every frame
    of the run that carries H or A and is not protected (F2's frames) becomes D. A run holding
    no D or X frame at all is left alone. D and X frames are never touched. Where rule D2 starts
    a stoppage later inside such a run (the players still running), this writes D over the
    carried letters before that start too: P4 overrides rule D2's later start.
    """
    codes = pm.label_codes(labels)
    out = codes.copy()
    team = (codes == pm.H) | (codes == pm.A)
    free = team & ~np.asarray(protected, dtype=bool)
    stops = (codes == pm.D) | (codes == pm.X)
    lost = ~np.isfinite(tracking.ball[:, 0])
    for _, first, stop in _blocks(tracking):
        for a, b in _runs(lost[first:stop]):
            a, b = first + a, first + b
            if b - a <= 2 * window or not stops[a:b].any():
                continue
            part = out[a:b]
            part[free[a:b]] = pm.D
    changed = out != codes
    return np.array(LABELS)[out].astype(np.asarray(labels).dtype), changed


def islands(labels, tracking, protected, fps=FPS):
    """P5 over a label array: (new labels, mask of the frames written D). Pure, like control().

    Inside each period, the spells are read the scorer's way (pm.fill_abstentions, then
    pm.segments). A spell of H or A shorter than dead_ball.MIN_OUT_S, with a D or the period's
    edge straight before it AND straight after it, and holding no protected frame (F2's), has
    its H and A frames written D; its X frames stay X (the scorer then reads them as the D
    before them).
    """
    codes = pm.label_codes(labels)
    out = codes.copy()
    shortest = int(round(MIN_OUT_S * fps))
    guarded = _counter(np.asarray(protected, dtype=bool))
    for _, first, stop in _blocks(tracking):
        run = [(first, stop)]
        segments = pm.segments(pm.fill_abstentions(codes, run), run)
        for i, (a, b, k) in enumerate(segments):
            if k not in pm.POSSESSION_LABELS or b - a >= shortest or guarded(a, b):
                continue
            before = segments[i - 1][2] if i > 0 else pm.D            # the period's edge
            after = segments[i + 1][2] if i + 1 < len(segments) else pm.D
            if before == pm.D and after == pm.D:
                part = out[a:b]
                part[part == k] = pm.D
    changed = out != codes
    return np.array(LABELS)[out].astype(np.asarray(labels).dtype), changed


def taker(tracking, frame):
    """The slot of the tracked dot nearest the ball on `frame` if within TAKER_M of it, else None
    (the ball untracked, or nobody that close)."""
    length, width = tracking.pitch
    ball, dots = tracking.ball[frame], tracking.players[frame]
    d = np.hypot((dots[:, 0] - ball[0]) * length, (dots[:, 1] - ball[1]) * width)
    d = np.where(np.isnan(d), np.inf, d)
    if len(d) == 0 or not np.isfinite(d.min()) or d.min() > TAKER_M:
        return None
    return int(np.argmin(d))


VOTER = {pm.H: HOME, pm.A: AWAY}                         # a team's letter -> its voter code


def _voted(tracking, votes, first, stop, code, skip):
    """Offsets from `first` of the frames of [first, stop) where the voter names the team of
    `code` (votes == VOTER[code]) through a dot that is not slot `skip`: on those frames the
    tracked dot nearest the ball (the first slot on an exact tie) is not `skip`. skip None: every
    such frame."""
    hits = np.asarray(votes)[first:stop] == VOTER[code]
    if skip is not None and hits.any():
        length, width = tracking.pitch
        ball, dots = tracking.ball[first:stop], tracking.players[first:stop]
        distance = np.hypot((dots[:, :, 0] - ball[:, None, 0]) * length,
                            (dots[:, :, 1] - ball[:, None, 1]) * width)
        distance = np.where(np.isnan(distance), np.inf, distance)
        hits &= distance.argmin(axis=1) != skip
    return np.flatnonzero(hits)


def restart_team(labels, tracking, restarts, protected, fps=FPS, reach_m=None, touch_m_s=None,
                 votes=None):
    """P3 over a label array: (new labels, mask of the frames rewritten). Pure, like control().

    "A corner, a goal kick or a free kick is taken by one team. From the kick that team has the
    ball, whatever our letter says, until a player of the other team gets to the ball: until he
    touches it, or until he is the player near the ball, whichever comes first."

    restarts = [(kick frame, team letter "H" or "A")], read from the labels as given and applied
    in frame order; a restart outside every period changes nothing. The other team's TOUCH is
    P1's: the ball's velocity changes by touch_m_s while a dot of that team that is not the
    taker's (taker() at the kick) is within reach_m. votes = per frame, the base's voter
    (pipeline.near_team: HOME / AWAY for the team of the tracked dot nearest the ball when it is
    within 3 m of it and the ball is not faster than 8 m/s, else nobody), or None.

    With votes (the rule above): for each restart whose kick frame carries a team's letter (H
    or A; a kick on a D or an X changes nothing), the reach runs from the kick to the frame
    before the first D frame or the first protected frame (F2's) after it, never past the
    period's end (an X reads as the letter before it, the scorer's way). The end is the EARLIER
    of the first frame of the reach where the other team touches the ball and the first frame
    where the voter names the other team through a dot that is not the taker's (_voted); neither
    is counted in the first KICK_GUARD_S after the kick. With no end the reach is written whole.
    Every frame from the kick up to (not including) the end that carries the other team's letter
    takes the restart's team's; its own team's letters, D, X and protected frames never change.

    Without votes (None, the rule as first built): only a restart whose kick frame carries the
    OTHER team's letter acts, its reach is that team's spell (it also stops at a D or a protected
    frame), and the end is the other team's first touch alone.
    """
    reach_m = REACH_M if reach_m is None else reach_m
    touch_m_s = TOUCH_M_S if touch_m_s is None else touch_m_s
    codes = pm.label_codes(labels)
    out = codes.copy()
    guard = np.asarray(protected, dtype=bool)
    if restarts:
        runs = [(first, stop) for _, first, stop in _blocks(tracking)]
        filled = pm.fill_abstentions(codes, runs)
        change = np.nan_to_num(ball_change(tracking, fps), nan=-1.0) >= touch_m_s
        length, width = tracking.pitch
        guard_after = int(round(KICK_GUARD_S * fps))
    for kick, letter in sorted(restarts or []):
        k = LABELS.index(letter)
        block = next(((first, stop) for first, stop in runs if first <= kick < stop), None)
        if block is None:
            continue
        if votes is None:
            if codes[kick] != 1 - k:
                continue
            ends = np.flatnonzero((filled[kick:block[1]] != 1 - k) | guard[kick:block[1]])
        else:
            if codes[kick] not in pm.POSSESSION_LABELS:
                continue
            ends = np.flatnonzero((filled[kick:block[1]] == pm.D) | guard[kick:block[1]])
        stop = kick + (int(ends[0]) if len(ends) else block[1] - kick)
        taken = taker(tracking, kick)
        slots = np.flatnonzero(tracking.team == (AWAY if k == pm.H else HOME))
        slots = slots[slots != taken]
        ball, dots = tracking.ball[kick:stop], tracking.players[kick:stop][:, slots]
        with np.errstate(invalid="ignore"):
            near = (np.hypot((dots[:, :, 0] - ball[:, None, 0]) * length,
                             (dots[:, :, 1] - ball[:, None, 1]) * width) <= reach_m).any(axis=1)
        touched = np.flatnonzero(change[kick:stop] & near)
        found = [int(f) for f in touched[touched >= guard_after][:1]]   # not the kick's own change
        if votes is not None:
            voted = _voted(tracking, votes, kick, stop, 1 - k, taken)
            found += [int(f) for f in voted[voted >= guard_after][:1]]
        stop = kick + min(found) if found else stop
        part = out[kick:stop]
        part[part == 1 - k] = k
    changed = out != codes
    return np.array(LABELS)[out].astype(np.asarray(labels).dtype), changed


def change_on_arrival(labels, tracking, votes, protected, window):
    """The change of team written on the ball's arrival, over a label array: (new labels, mask
    of the frames rewritten). Pure, like control().

    "A change of team is written where the new team first has the ball near. The vote changes
    team in the middle of a pass; the possession changes when the new team gets the ball."

    votes = per frame, the voter the base reads (pipeline.near_team: HOME / AWAY for the team of
    the nearest dot when it is near the ball and the ball is not in flight, else nobody).
    Inside each period, the spells are read the scorer's way (pm.fill_abstentions, then
    pm.segments). For every spell [a, b) of one team that follows a spell of the other team
    straight away (no dead ball between them), f is the first frame of a .. min(b - 1, a +
    window) where the voter names the new team. If there is one, the frames a .. a + d - 1, with
    d = min(f - a, (b - a) // 2), that carry the new team's letter and are not protected (F2's
    frames) take the team before. If there is none, nothing changes. A change is only ever moved
    later, never earlier, and by at most half the new spell, so no spell is added or removed,
    with one exception: a frame F2 wrote inside the moved stretch keeps its letter, and if it
    is not the stretch's last frame the spell is split there (once on one of the three sample
    games, a spell of one frame; never on our match). D and X frames are never touched.
    """
    codes = pm.label_codes(labels)
    out = codes.copy()
    votes = np.asarray(votes)
    guard = np.asarray(protected, dtype=bool)
    for _, first, stop in _blocks(tracking):
        run = [(first, stop)]
        segments = pm.segments(pm.fill_abstentions(codes, run), run)
        for i in range(1, len(segments)):
            a, b, k = segments[i]
            if k not in pm.POSSESSION_LABELS or segments[i - 1][2] != 1 - k:
                continue
            hits = np.flatnonzero(votes[a:min(b, a + window + 1)] == VOTER[k])
            if not len(hits):
                continue
            d = min(int(hits[0]), (b - a) // 2)
            part = out[a:a + d]
            part[(codes[a:a + d] == k) & ~guard[a:a + d]] = 1 - k
    changed = out != codes
    return np.array(LABELS)[out].astype(np.asarray(labels).dtype), changed


__all__ = ["SHORT_S", "REACH_M", "TOUCH_M_S", "MERGE_BEFORE_DEAD", "TAKER_M", "ball_change",
           "within_reach", "touches", "control", "gap_edges", "islands", "taker",
           "restart_team", "change_on_arrival"]
