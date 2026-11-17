"""When the ball is dead: the rules that write D over the smoothed labels.

v0 calls the ball dead only where the tracking has lost it for a while, which is "the tracker is
blind", not "the ball is out of play". These rules decide D from evidence instead. Each returns a
boolean mask over video frames (True = dead), and pipeline.label writes D on those frames AFTER
the +/- 50-frame smoothing, never before: smoothing a D stretch would blur its edges by up to 2 s,
and the edges are where the score is (EVALUATION.md: "perfect but 1 s late" = 0.6539).

The rules, in plain words (the wording the team reads):

  B  Before kickoff. Every graded frame before a half's kickoff is D: play has not started.
     (The second half is graded from 110808 but kicks off at 113742, FAQ.md.)

  B2 Outside the halves. In the team's words: "outside the two halves (before the match,
     halftime, after the final whistle) the ball is not in play, so every such frame is D."
     These frames are never graded; the rule keeps any view of the whole video from showing
     possessions while players warm up or walk off.

  C  Ball off the pitch. The ball is out when it is wholly over a touchline or a goal line (Law 9):
     its centre is past the line by at least the ball's radius, 11 cm. It is D from the first
     frame it is out until the frame before it is next tracked back inside the pitch (or the end
     of the half). A ball back inside within 1 second was never out: nobody restarts play that
     fast, a player first has to collect the ball, so a short blip over the line is tracking
     jitter. For the same reason two out stretches less than 1 second apart are one stoppage.

  C2 Out while the tracker had lost it. The tracking has no height, so a ball in the air is often
     lost. If the ball is next seen already off the pitch, it crossed the line during the gap.
     The crossing is estimated by carrying the ball on from its last tracked position at its last
     tracked speed and direction until it is wholly over a line. If that cannot be done (no speed
     known, the ball standing still, or the estimate later than the moment the ball was seen out:
     it must have been played again in the gap), the stoppage starts where the tracker lost the
     ball.

  The restart test (the team's words). "The restart is the ball coming into play and at
     least one of: (a) one or more players accelerate towards the ball; (b) players further away
     speed up at the same moment, on both teams; (c) two opposing players are around the ball,
     contesting it. A fast ball with nobody moving (a pass to the taker, a ball boy's throw) is
     not the restart." In the code: the ball is kicked (4 m/s or more, as below), and (a) a
     player within 15 m of the ball speeds up (his mean speed over the next 2 seconds is at least
     0.5 m/s above the last second) and closes on the ball, or (b) at least one player of each
     team further than 15 m from the ball speeds up the same way. (c) is not used: on the
     practice games two opponents are within 3 m of the ball at only 6 restarts in 100 (players
     keep 9.15 m away at a set piece, Law 13), but at every fast ball nobody played. The players
     are only read inside the kick's half: halftime never counts.

  D  Restarts from the Smart Tagging. A set-piece tag (throw-in, corner, goal kick, free kick,
     kick-off after a goal) says a restart happens around then; it usually starts while the ball
     is still dead. Every set-piece tag is treated alike: the tag's type can be wrong, so it is
     only a hint. The restart is the first kick at or after the tag's start, inside the tag (the
     ball tracked inside the pitch, moving away at 4 m/s or more and still travelling that fast
     0.2 s later). A later kick replaces it only when the restart test above holds for it within
     1.6 seconds of the first (the players react a moment late, or the first was a pass to the
     taker nobody reacted to); otherwise the first kick stays the restart, so a restart the test
     misses never stretches D into the live play after it. The stoppage starts
     where the ball was lost or went out after it was last kicked before the tag (rule C2's
     estimate when the ball came back already off the pitch). It is D from there to the frame
     before the restart. A tag with no ball loss before its restart, or with no restart inside
     it, writes nothing: a tag is evidence, not an answer. The kick-off at the start of a half is
     rule B's.

  D2 Where a restart's stoppage starts, from the players (the team's words). "The ball went
     out when the players nearest to it suddenly slow down and the players everywhere else on
     the pitch are slow too." When rule D's stoppage starts at a lost ball
     (not one seen over a line) while both teams are still clearly running (the slower team's
     mean speed over 4 seconds at least 1.54 m/s), play was still going on: the stoppage starts
     where the players slow down below that, less the second they take to slow down. Both teams
     must be seen: if either team's speed is unknown, the players say nothing, and a moment with
     the players unknown is never read as "slowed down". The 4 seconds stay inside the half.

  D' Goals. After a goal it is D from the moment the ball is wholly over the goal line until the
     kick-off after the goal is taken. The goal is the first time the ball is seen over a goal
     line after the goal tag starts (and after any other restart before that kick-off).
     "When no kick-off can be seen in the tracking, the kick-off tag's start is the restart"
     (no kick inside the kick-off tag; GOAL_TAG_RESTART).

  F3 A huddle (the team's words). "Many players of both teams bunched in one small area
     while the ball is away from them is a stoppage (a confrontation, a card)." While the ball is
     not tracked: at least 6 players, 2 or more of each team, within 5 m of one player (a circle
     5 m around him), walking slowly (their mean speed 1 m/s or less), for at least 1 second.
     It is D while the huddle lasts. A corner's crowded box is not a huddle: players move fast
     there.

The lost-ball corrections come last, after every D above, and never undo any of it (B, B2, C,
D, D2, D', F3): with the ball lost, a tagged stoppage or a huddle outranks a guess that play went
on:

  F1 A ball in flight (the team's words). "A ball the tracker loses right after a fast kick
     is in the air: play is live until it comes down, unless it next shows up off the pitch."
     The tracking has no height, so a ball in the air is often lost. When the ball is lost within
     1 second of a kick, still travelling at kick speed (4 m/s), and is next seen inside the pitch
     within 4 seconds (the longest a kicked ball stays up), play went on: those frames are X, not
     D, and the scorer carries the team that had the ball over them. Next seen off the pitch:
     rule C2 decides. Lost for longer: left as it is.

  F2 The goalkeeper holding the ball (the team's words; Law 12). "A ball in the goalkeeper's
     hands is in play: the keeper's team has it until he releases it, even while the tracker
     cannot see it." The keeper is his team's player nearest its own goal. When the ball is lost
     within 3 m of him inside his own penalty area (Law 1: 16.5 m deep, 40.32 m wide), and is next
     seen inside the pitch moving away from him (his kick or throw) within 17.7 seconds (on the
     practice games 9 catches in 10 are released within 13.7 seconds; in the
     team's words: "a goalkeeper often keeps the ball at his feet until an opponent presses
     him"; plus the 4 seconds a kicked ball can stay up), those frames are his team's.

The numbers are football knowledge (Laws of the Game), checked against the practice games'
events, or measured on the practice games as physical quantities, never read off the competition
match. Nothing here reads files or prints.
"""

import numpy as np

from common import FPS
from tracking import AWAY, HOME, NOBODY

# Params
BALL_RADIUS_M = 0.11   # Law 2: circumference 68-70 cm, so the radius is 10.8-11.1 cm
MIN_OUT_S = 1.0        # no restart is taken within 1 s of the ball leaving the pitch
TOLERANCE = 1e-9       # float rounding: a ball exactly one radius over the line is out
# A restart is taken when the ball "is kicked and clearly moves" (Laws 13-17). Faster than a
# player jogging with the ball in hand (about 3 m/s), slower than the softest restart pass.
MOVING_M_S = 4.0
MOVING_S = 0.2         # and still travelling that fast this long after: a tracking jump is not a kick
# Smart Tagging names, matched case-insensitively as the starter loader does (DATA.md tag table).
SET_PIECES, GOALS = "set pieces", "goals"
KICK_OFF_START, KICK_OFF_AFTER_GOAL = "kick off (start)", "kick off (after goal)"
# F1: a hard lofted kick (about 28 m/s at 45 degrees) stays up 2 v sin45 / g = 4.0 s; on the
# sample games no ball lost after a kick came back inside between 3 and 4 s.
FLIGHT_S = 4.0
LOST_AFTER_KICK_S = 1.0  # "right after a kick": the ball still in that kick's flight
# F2: Law 1 penalty area; a keeper's reach with a step (sample games: ball <= 3.7 m from him at a
# save); the hold, then the release's flight (FLIGHT_S). The hold is the 90th percentile of the
# sample games' release times (a keeper's recovery in his box to his team's next event, 66
# catches; median 3.8 s, p90 13.7 s), tuned on outside data only; Law 12
# says six seconds, often not enforced.
BOX_DEPTH_M, BOX_WIDTH_M = 16.5, 40.32
KEEPER_REACH_M = 3.0
HOLD_S = 13.7
# The restart test, measured on the sample games: at 240 restarts vs 1150
# dead moments with the ball tracked, "(a) or (b)" with a gain of 0.5 m/s passes 91.7% of the
# restarts and 27.8% of the dead moments (82.8% of live kicks).
GAIN_M_S = 0.5           # a player speeds up: his mean speed after beats before by this much
BEFORE_S, AFTER_S = 1.0, 2.0   # before and after the kick (at a restart everyone speeds up
                               # over the next 2 s, on the sample games)
NEAR_BALL_M = 15.0       # (a) near the ball; (b) further away
CHASE_S = 1.0            # (a) closing on the ball over this time
# The restart is the first kick inside the tag; a later kick replaces it only if it passes the
# restart test within this time. On the sample games 19 of 241 restarts fail the test
# on their first frame and a frame that passes follows a median 0.32 s later, p90 1.57 s (16 of
# the 19 within the same kick); a fake kick before a restart showed at 1 of 225, too few to measure.
REPLACE_S = 1.6
# D': the fallback of goal_stoppage. It is switched on only if it turns at least one H or A frame
# into D on the competition match (no stage is kept that changes nothing; "does it change
# anything" is not a tuned number). Counted on 2026-11-14, the built goal_stoppage called with
# the switch:
# no kick inside the kick-off tag: the tag's start is the restart. It changes letters: ON.
GOAL_TAG_RESTART = True
# D2: the slower team's mean speed over 4 s (centred). On the sample games 90% of dead frames are
# under 1.54 m/s (73% of live frames reach it); after the ball goes dead that signal drops below
# 1.54 a median 0.96 s later (219 dead onsets with the players clearly live 3 s before).
TEAM_WINDOW_S = 4.0
LIVE_BAR_M_S = 1.54
SLOWDOWN_LAG_S = 1.0
# F3: within 5 m of one player (a circle 5 m around him); 6 players, 2 of each team (a card's
# huddle on the sample games: median 6); walking slowly (sport science: walking is under
# 2 m/s; huddles at sample cards move at a median 0.8 m/s, a corner's delivery at 1.9 m/s).
HUDDLE_RADIUS_M = 5.0
HUDDLE_SIZE, HUDDLE_EACH = 6, 2
HUDDLE_SPEED_M_S = 1.0
SPEED_HALF_S = 0.2       # a player's speed: displacement over +/- 0.2 s (as measured)
CHUNK = 2000             # frames per vectorised step of the huddle search (about 35 MB)


def _blocks(tracking):
    """The periods of play; a Tracking without blocks is one period over all its frames."""
    return tracking.blocks or [(None, 0, tracking.n_frames)]


def before_kickoff(tracking):
    """Rule B: True on the graded frames of a half before its kickoff.

    A Tracking without kickoffs (the practice games) has no frame before kickoff here.
    """
    dead = np.zeros(tracking.n_frames, dtype=bool)
    for half, first, stop in _blocks(tracking):
        kickoff = (tracking.kickoffs or {}).get(half)
        if kickoff is not None:
            dead[first:max(first, min(stop, kickoff))] = True
    return dead


def outside_the_halves(tracking):
    """Rule B2: True on every frame outside the periods of play (before the match, halftime,
    after the final whistle). A Tracking without blocks is all one period: nothing marked.
    """
    dead = np.ones(tracking.n_frames, dtype=bool)
    for _, first, stop in _blocks(tracking):
        dead[first:stop] = False
    return dead


def metres_over_the_line(ball, pitch):
    """Per frame, how far in metres the ball's centre is past the line it is furthest over.

    Zero or negative while the centre is on or inside every line; NaN where the ball is not
    tracked. ball is (n, 2), normalised 0-1 to the pitch's (length, width).
    """
    length, width = pitch
    x, y = ball[:, 0] * length, ball[:, 1] * width
    return np.max(np.stack([-x, x - length, -y, y - width]), axis=0)


def wholly_out(ball, pitch):
    """True where the tracked ball is wholly over a touchline or goal line (Law 9)."""
    over = np.nan_to_num(metres_over_the_line(ball, pitch), nan=-np.inf)
    return over >= BALL_RADIUS_M - TOLERANCE


def crossing_frame(ball, last, pitch):
    """Rule C2: the first frame after `last` at which the ball, carried on from frame `last` at
    the velocity it had between frames last - 1 and last, is wholly over a line.

    None when that cannot be estimated: either of the two frames untracked, or the ball still.
    """
    if last < 1 or not np.all(np.isfinite(ball[last - 1:last + 1])):
        return None
    length, width = pitch
    scale = np.array([length, width])
    position = ball[last] * scale
    velocity = (ball[last] - ball[last - 1]) * scale  # metres per frame
    steps = []
    for axis, size in ((0, length), (1, width)):
        if velocity[axis] == 0:
            continue
        target = size + BALL_RADIUS_M if velocity[axis] > 0 else -BALL_RADIUS_M
        steps.append(int(np.ceil((target - position[axis]) / velocity[axis] - TOLERANCE)))
    if not steps:
        return None
    return last + max(1, min(steps))


def _out_stretches(out, inside):
    """[(first out frame, first inside frame after it)] for each run of 'not back inside yet'.

    A stretch starts at an out frame and runs through any untracked or out frames until the ball
    is tracked inside again; len(out) when it never is.
    """
    out_at = np.flatnonzero(out)
    if len(out_at) == 0:
        return []
    inside_at = np.flatnonzero(inside)
    following = np.searchsorted(inside_at, out_at)
    back_in = np.full(len(out_at), len(out), dtype=np.int64)
    found = following < len(inside_at)
    back_in[found] = inside_at[following[found]]
    new = np.concatenate(([True], back_in[1:] != back_in[:-1]))
    return list(zip(out_at[new].tolist(), back_in[new].tolist()))


def _stoppage_start(ball, first_out, pitch):
    """Where the stoppage that ends with the ball out at `first_out` begins (rules C and C2)."""
    if first_out == 0 or np.isfinite(ball[first_out - 1, 0]):
        return first_out                       # seen crossing: the ball was inside just before
    tracked = np.flatnonzero(np.isfinite(ball[:first_out, 0]))
    if len(tracked) == 0:
        return 0                               # lost since the start of the period
    last = int(tracked[-1])
    crossing = crossing_frame(ball, last, pitch)
    if crossing is None or crossing > first_out:
        return last + 1                        # the first frame of the gap
    return crossing


def off_the_pitch(tracking, fps=FPS):
    """Rule C (with C2): True from the ball going wholly out until the frame before it is back.

    Stretches less than MIN_OUT_S apart are merged into one stoppage, then any stoppage shorter
    than MIN_OUT_S is dropped, so no D written here is shorter than 1 s.
    """
    dead = np.zeros(tracking.n_frames, dtype=bool)
    min_frames = int(round(MIN_OUT_S * fps))
    for _, first, stop in _blocks(tracking):
        ball = tracking.ball[first:stop]
        out = wholly_out(ball, tracking.pitch)
        inside = np.isfinite(ball[:, 0]) & ~out
        merged = []
        for first_out, back_in in _out_stretches(out, inside):
            start = _stoppage_start(ball, first_out, tracking.pitch)
            if merged and start - merged[-1][1] < min_frames:
                merged[-1][1] = back_in
            else:
                merged.append([start, back_in])
        for start, end in merged:
            if end - start >= min_frames:
                dead[first + start:first + end] = True
    return dead


def kicked(ball, pitch, fps=FPS):
    """True on frame f where the ball is kicked: tracked inside the pitch on f and the next
    MOVING_S, moving at MOVING_M_S or more from f to f + 1, on average from f to the end of that
    time, and on average from f + 1 to the end (it keeps going after the first step).

    A ball standing still, rolled slowly, or jumping once (a tracking error) is not kicked.
    """
    n = len(ball)
    k = max(2, int(round(MOVING_S * fps)))
    out = np.zeros(n, dtype=bool)
    if n <= k:
        return out
    inside = np.isfinite(ball[:, 0]) & ~wholly_out(ball, pitch)
    count = np.concatenate(([0], np.cumsum(inside)))
    xy = ball * np.array(pitch)
    f = np.arange(n - k)

    def speed(a, b, frames):
        return np.nan_to_num(np.hypot(*(xy[b] - xy[a]).T) * fps / frames)

    out[f] = ((count[f + k + 1] - count[f] == k + 1) &
              (speed(f, f + 1, 1) >= MOVING_M_S) &
              (speed(f, f + k, k) >= MOVING_M_S) &
              (speed(f + 1, f + k, k - 1) >= MOVING_M_S))
    return out


def window_means(values, ok, lo, hi):
    """Per frame f, the mean of values over frames f + lo .. f + hi - 1 where ok (along axis 0);
    NaN where fewer than half of those frames are ok."""
    v = np.where(ok, np.nan_to_num(values), 0.0)
    zero = np.zeros((1,) + v.shape[1:])
    total = np.concatenate((zero, np.cumsum(v, axis=0)))
    count = np.concatenate((zero, np.cumsum(ok, axis=0)))
    f = np.arange(len(values))
    a, b = np.clip(f + lo, 0, len(values)), np.clip(f + hi, 0, len(values))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (total[b] - total[a]) / (count[b] - count[a])
    out[(count[b] - count[a]) < (hi - lo) / 2.0] = np.nan
    return out


def played(tracking, kicks, speed=None, fps=FPS):
    """The restart test: True on the kick frames where the players show the ball was
    played: (a) a player within NEAR_BALL_M of the ball speeds up by GAIN_M_S and closes on it
    over CHASE_S, or (b) a player of each team further away speeds up by GAIN_M_S. Speeding up =
    mean speed over the next AFTER_S minus the last BEFORE_S. Only dots inside the pitch count,
    and every window stays inside the kick's half (a kick outside the halves is never played)."""
    speed = player_speeds(tracking, fps) if speed is None else speed
    out = np.zeros(tracking.n_frames, dtype=bool)
    if not kicks.any():
        return out
    ok = on_the_pitch(tracking.players) & np.isfinite(speed)
    scale = np.array(tracking.pitch)
    home, away = tracking.team == HOME, tracking.team == AWAY
    for _, first, stop in _blocks(tracking):    # every window stays inside its half
        frames = first + np.flatnonzero(kicks[first:stop])
        if len(frames) == 0:
            continue
        s, o, at = speed[first:stop], ok[first:stop], frames - first
        before = window_means(s, o, -int(round(BEFORE_S * fps)), 0)[at]
        after = window_means(s, o, 0, int(round(AFTER_S * fps)))[at]
        later = np.minimum(frames + int(round(CHASE_S * fps)), stop - 1)
        ball = tracking.ball[frames] * scale
        seen = np.isfinite(tracking.ball[later])
        target = np.where(seen, tracking.ball[later], tracking.ball[frames]) * scale
        now, then = tracking.players[frames] * scale, tracking.players[later] * scale
        with np.errstate(invalid="ignore"):
            up = ok[frames] & (after - before >= GAIN_M_S)
            near = np.hypot(*(now - ball[:, None]).transpose(2, 0, 1)) <= NEAR_BALL_M
            closing = (np.hypot(*(then - target[:, None]).transpose(2, 0, 1)) <
                       np.hypot(*(now - target[:, None]).transpose(2, 0, 1)))
        far = up & ~near
        out[frames] = ((up & near & closing).any(1) |
                       (far[:, home].any(1) & far[:, away].any(1)))
    return out


def slower_team_speed(tracking, speed=None, fps=FPS):
    """Per frame, the slower team's mean speed over TEAM_WINDOW_S (centred, never across a half's
    edge), dots inside the pitch only; NaN where unknown, and unknown when either team's speed
    is: one team alone never stands for both. NaN outside the halves."""
    speed = player_speeds(tracking, fps) if speed is None else speed
    ok = on_the_pitch(tracking.players) & np.isfinite(speed)
    half = int(round(TEAM_WINDOW_S * fps / 2))
    per_team = []
    for team in (HOME, AWAY):
        cols = tracking.team == team
        m = ok[:, cols]
        with np.errstate(invalid="ignore", divide="ignore"):
            per_team.append(np.where(m, speed[:, cols], 0.0).sum(1) / m.sum(1))
    out = np.full(tracking.n_frames, np.nan)
    for _, first, stop in _blocks(tracking):
        means = []
        for per_frame in per_team:
            v = per_frame[first:stop, None]
            means.append(window_means(v, np.isfinite(v), -half, half + 1)[:, 0])
        with np.errstate(invalid="ignore"):
            out[first:stop] = np.minimum(means[0], means[1])    # NaN if either is NaN
    return out


def slowdown_start(signal, begin, restart, fps=FPS):
    """Rule D2: where a stoppage that starts at a lost ball at `begin` really starts. If both
    teams are clearly running at `begin` (signal >= LIVE_BAR_M_S), the first frame before the
    restart where the signal is known and below that, less SLOWDOWN_LAG_S (never before
    `begin`); else `begin`. A frame with the players unknown (NaN) is skipped, never read as
    "slowed down". Also `begin` when they are not seen slowing down before the restart."""
    if signal is None or not signal[begin] >= LIVE_BAR_M_S:
        return begin
    with np.errstate(invalid="ignore"):
        below = np.flatnonzero(signal[begin:restart] < LIVE_BAR_M_S)
    if len(below) == 0:
        return begin
    return max(begin, begin + int(below[0]) - int(round(SLOWDOWN_LAG_S * fps)))


def _type(tag):
    return (tag["tags"].get("Type") or "").strip().lower()


def _code(tag):
    return (tag["code"] or "").strip().lower()


def _block_of(tracking, frame):
    """(first, stop) of the period of play holding `frame`, or None."""
    for _, first, stop in _blocks(tracking):
        if first <= frame < stop:
            return first, stop
    return None


def restart_frame(kicks, start, end, ready=None, fps=FPS):
    """Rules D and D': the restart inside a tag (from `start`, before `end`), or None.

    The first kick there is the restart. With `ready` (the kicks that pass the restart test,
    played), a later kick replaces it only when it passes the test within REPLACE_S of the first
    (a pass to the taker that nobody reacts to, then the real kick); otherwise the first kick
    stays, so a missed restart never stretches D into the live play after it.
    """
    stop = max(end, start + 1)
    hits = np.flatnonzero(kicks[start:stop])
    if len(hits) == 0:
        return None
    restart = start + int(hits[0])
    if ready is None:
        return restart
    passing = np.flatnonzero(ready[restart:min(stop, restart + int(round(REPLACE_S * fps)) + 1)])
    return restart + int(passing[0]) if len(passing) else restart


def lost_ball_start(tracking, kicks, block, start, restart, fps=FPS, signal=None):
    """Rule D: the first dead frame of the stoppage a tag starting at `start` points at, whose
    restart is `restart`; None when the ball is never lost before it.

    The ball was lost or went out after it was last kicked before `start` (inside the period
    `block` = (first, stop)). If it was lost and is next seen already off the pitch, rule C2's
    crossing estimate; if it was lost and is next seen inside, rule D2 asks the players (signal,
    the slower team's speed; None skips it).
    """
    first, stop = block
    ball = tracking.ball[first:stop]
    lost = ~np.isfinite(ball[:, 0]) | wholly_out(ball, tracking.pitch)
    before = np.flatnonzero(kicks[first:start])
    last_kick = int(before[-1]) if len(before) else 0
    losses = np.flatnonzero(lost[last_kick:restart - first])
    if len(losses) == 0:
        return None
    begin = last_kick + int(losses[0])
    if not np.isfinite(ball[begin, 0]):        # lost: does it come back already off the pitch?
        seen = np.flatnonzero(np.isfinite(ball[begin:, 0]))
        back = begin + int(seen[0]) if len(seen) else None
        if back is not None and lost[back]:
            begin = _stoppage_start(ball, back, tracking.pitch)
        elif signal is not None:               # rule D2: lost, not seen out: ask the players
            begin = slowdown_start(signal, first + begin, restart, fps) - first
    return first + begin


def set_piece_stoppage(tracking, kicks, tag, fps=FPS, ready=None, signal=None):
    """Rule D for one set-piece tag: (first dead frame, restart frame), or None.

    kicks: every kick (kicked), used for the last kick before the tag and the first kick inside
    it; ready: the kicks that pass the restart test (played), used by restart_frame (None: the
    first kick is the restart); signal: the slower team's speed (slower_team_speed) for rule D2,
    or None to skip it.
    """
    start = tag["start_frame"]
    block = _block_of(tracking, start)
    if block is None:
        return None
    restart = restart_frame(kicks, start, min(tag["end_frame"], block[1]), ready, fps)
    if restart is None:
        return None
    begin = lost_ball_start(tracking, kicks, block, start, restart, fps, signal)
    if begin is None or restart - begin < int(round(MIN_OUT_S * fps)):
        return None
    return begin, restart


def goal_stoppage(tracking, kicks, goal, tags, fps=FPS, ready=None, tag_restart=None):
    """Rule D' for one GOALS tag: (the frame the ball is over the goal line, kick-off restart),
    or None when there is no kick-off after it in the half, or no goal-line frame before it.
    kicks and ready as in set_piece_stoppage: the restart is chosen by restart_frame.

    The fallback (the wording the team reads): "After a goal the ball is dead until the
    kick-off. When no kick-off can be seen in the tracking, the kick-off tag's start is the
    restart." tag_restart (default GOAL_TAG_RESTART, read when called): no kick inside the
    kick-off tag: the restart is that tag's start frame. The 1 s floor holds.
    """
    tag_restart = GOAL_TAG_RESTART if tag_restart is None else tag_restart
    block = _block_of(tracking, goal["start_frame"])
    if block is None:
        return None
    first, stop = block
    later = sorted((t for t in tags if goal["start_frame"] <= t["start_frame"] < stop),
                   key=lambda t: t["start_frame"])
    kick_off = next((t for t in later if _code(t) == SET_PIECES and
                     _type(t) == KICK_OFF_AFTER_GOAL), None)
    if kick_off is None or any(_code(t) == GOALS and t["start_frame"] < kick_off["start_frame"]
                               for t in later if t is not goal):
        return None                            # no kick-off, or it belongs to a later goal
    restart = restart_frame(kicks, kick_off["start_frame"], min(kick_off["end_frame"], stop),
                            ready, fps)
    if restart is None and tag_restart:
        restart = kick_off["start_frame"]      # no kick-off seen: the tag's start
    if restart is None:
        return None
    since = max([goal["start_frame"]] + [t["start_frame"] for t in later
                if _code(t) == SET_PIECES and t["start_frame"] < kick_off["start_frame"]
                and t is not kick_off])
    length = tracking.pitch[0]
    x = tracking.ball[since:restart, 0] * length
    beyond = np.nan_to_num(np.maximum(-x, x - length), nan=-np.inf)
    over = np.flatnonzero(beyond >= BALL_RADIUS_M - TOLERANCE)
    if len(over) == 0 or restart - (since + int(over[0])) < int(round(MIN_OUT_S * fps)):
        return None
    return since + int(over[0]), restart


def restarts(tracking, tags, fps=FPS):
    """Rules D, D2 and D': True from each tagged stoppage's start to the frame before its
    restart."""
    dead = np.zeros(tracking.n_frames, dtype=bool)
    for _, stretch in restart_stoppages(tracking, tags, fps):
        if stretch is not None:
            dead[stretch[0]:stretch[1]] = True
    return dead


def restart_stoppages(tracking, tags, fps=FPS):
    """[(tag, (first dead frame, restart frame) or None)] for every SET PIECES (but the first
    kick-off of a half, rule B's) and GOALS tag: rules D, D2 and D'."""
    kicks = kicked(tracking.ball, tracking.pitch, fps)
    speed = player_speeds(tracking, fps)
    ready = played(tracking, kicks, speed, fps)
    signal = slower_team_speed(tracking, speed, fps)
    out = []
    for tag in tags:
        if _code(tag) == SET_PIECES and _type(tag) != KICK_OFF_START:
            stretch = set_piece_stoppage(tracking, kicks, tag, fps, ready, signal)
        elif _code(tag) == GOALS:
            stretch = goal_stoppage(tracking, kicks, tag, tags, fps, ready)
        else:
            continue
        out.append((tag, stretch))
    return out


def _runs(mask):
    """[(start, stop)] of the True runs of a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def ball_gaps(ball, first, stop):
    """[(a, b)] of the ball gaps inside [first, stop) with the ball seen on both sides: frames
    a .. b - 1 untracked, a - 1 its last sighting, b the frame it is seen again."""
    lost = ~np.isfinite(ball[first:stop, 0])
    return [(first + a, first + b) for a, b in _runs(lost) if a > 0 and b < stop - first]


def on_the_pitch(players):
    """True where a player's dot is tracked inside the pitch. Substitutes warming up, staff and
    anyone by the touchline stand outside it, so the players' rules never count them."""
    x, y = players[..., 0], players[..., 1]
    with np.errstate(invalid="ignore"):
        return (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)


def player_speeds(tracking, fps=FPS):
    """(n, s) each player's speed in m/s, his displacement over +/- SPEED_HALF_S, never across a
    half's edge; NaN if unknown (and outside the halves)."""
    h = max(1, int(round(SPEED_HALF_S * fps)))
    xy = tracking.players * np.array(tracking.pitch)
    speed = np.full(xy.shape[:2], np.nan)
    for _, first, stop in _blocks(tracking):
        if stop - first > 2 * h:
            step = xy[first + 2 * h:stop] - xy[first:stop - 2 * h]
            speed[first + h:stop - h] = np.hypot(step[..., 0], step[..., 1]) * fps / (2.0 * h)
    return speed


def huddle_frames(tracking, frames, speed):
    """Rule F3's test on each frame in `frames`: the biggest circle of HUDDLE_RADIUS_M around a
    player that holds players of both teams has HUDDLE_SIZE or more players, HUDDLE_EACH or more
    of each team, and their mean speed is HUDDLE_SPEED_M_S or less (a player with no known speed
    counts as fast). Only dots inside the pitch count."""
    xy = tracking.players * np.array(tracking.pitch)
    home_slot, away_slot = tracking.team == HOME, tracking.team == AWAY
    out = np.zeros(len(frames), dtype=bool)
    for c in range(0, len(frames), CHUNK):
        f = frames[c:c + CHUNK]
        p, ok = xy[f], on_the_pitch(tracking.players[f])
        d = np.hypot(p[:, :, None, 0] - p[:, None, :, 0], p[:, :, None, 1] - p[:, None, :, 1])
        with np.errstate(invalid="ignore"):
            inside = (d <= HUDDLE_RADIUS_M) & ok[:, :, None] & ok[:, None, :]
        home, away = inside[:, :, home_slot].sum(2), inside[:, :, away_slot].sum(2)
        size = np.where((home >= 1) & (away >= 1), home + away, 0)
        rows, best = np.arange(len(f)), np.argmax(size, axis=1)
        members = inside[rows, best]
        v = np.where(np.isfinite(speed[f]), speed[f], np.inf)
        mean_v = np.where(members, v, 0.0).sum(1) / np.maximum(1, members.sum(1))
        out[c:c + len(f)] = ((size[rows, best] >= HUDDLE_SIZE) &
                             (np.minimum(home[rows, best], away[rows, best]) >= HUDDLE_EACH) &
                             (mean_v <= HUDDLE_SPEED_M_S))
    return out


def huddle(tracking, fps=FPS):
    """Rule F3: True while a huddle of both teams stands with the ball not tracked, 1 s or more."""
    dead = np.zeros(tracking.n_frames, dtype=bool)
    min_frames = int(round(MIN_OUT_S * fps))
    speed = None
    for _, first, stop in _blocks(tracking):
        lost = first + np.flatnonzero(~np.isfinite(tracking.ball[first:stop, 0]))
        if len(lost) < min_frames:
            continue
        speed = player_speeds(tracking, fps) if speed is None else speed
        hit = np.zeros(tracking.n_frames, dtype=bool)
        hit[lost] = huddle_frames(tracking, lost, speed)
        for a, b in _runs(hit):
            if b - a >= min_frames:
                dead[a:b] = True
    return dead


def in_flight(tracking, fps=FPS):
    """Rule F1: True on each ball gap that a ball in flight explains: lost within
    LOST_AFTER_KICK_S of a kick, still at kick speed when last seen inside the pitch, and seen
    again inside the pitch within FLIGHT_S.

    The program writes X only where such a gap is still D. With the +/- 2 s vote before it, a
    gap of FLIGHT_S or less is reached on every frame by the votes from its two sides, so unless
    no player is tracked there it never holds a D for this rule to change: it has written no
    frame on any match we hold. It is kept as written."""
    ball, pitch = tracking.ball, tracking.pitch
    live = np.zeros(tracking.n_frames, dtype=bool)
    kicks = kicked(ball, pitch, fps)
    kicks = kicks & ~np.concatenate(([False], kicks[:-1]))   # the frame each kick starts from
    out = wholly_out(ball, pitch)
    xy = ball * np.array(pitch)
    after, flight = int(round(LOST_AFTER_KICK_S * fps)), int(round(FLIGHT_S * fps))
    for _, first, stop in _blocks(tracking):
        for a, b in ball_gaps(ball, first, stop):
            last = a - 1
            if b - a > flight or out[b] or out[last] or last - 1 < first or \
                    not np.isfinite(ball[last - 1, 0]):
                continue
            if np.hypot(*(xy[last] - xy[last - 1])) * fps < MOVING_M_S:
                continue
            if kicks[max(first, a - after):a].any():
                live[a:b] = True
    return live


def own_goal_x(tracking, team, first, stop):
    """0.0 or 1.0, the goal line a team defends in a period (the side its dots stand on, on
    average); None if none of its dots is tracked there."""
    x = tracking.players[first:stop][:, tracking.team == team, 0]
    if not np.isfinite(x).any():
        return None
    return 0.0 if np.nanmean(x) < 0.5 else 1.0


def keeper(tracking, frame, team, goal):
    """The (x, y) of the team's dot nearest the centre of its own goal on `frame` (dots inside the
    pitch only): its goalkeeper. None if the team has no dot on the pitch."""
    length, width = tracking.pitch
    p = tracking.players[frame, tracking.team == team]
    ok = on_the_pitch(p)
    if not ok.any():
        return None
    d = np.where(ok, np.hypot((p[:, 0] - goal) * length, (p[:, 1] - 0.5) * width), np.inf)
    return p[int(np.argmin(d))]


def in_own_box(xy, goal, pitch):
    """True where the normalised point xy is inside the penalty area at goal line `goal` (Law 1)."""
    length, width = pitch
    return (abs(xy[0] - goal) * length <= BOX_DEPTH_M and
            abs(xy[1] - 0.5) * width <= BOX_WIDTH_M / 2)


def keeper_holding(tracking, fps=FPS):
    """Rule F2: per frame the team (HOME, AWAY) whose goalkeeper holds the lost ball, else NOBODY.

    The ball is last seen inside the team's own penalty area within KEEPER_REACH_M of its keeper,
    and is seen again inside the pitch within HOLD_S + FLIGHT_S, moving away from the keeper at
    MOVING_M_S or more over the next MOVING_S (his kick or throw).

    The two teams defend opposite goals. Each team's goal is read from where its own players
    stand (own_goal_x), one team at a time, so a team that spends a period camped in the other
    half can be given the same goal as its opponent. In a period where the two teams' goals do
    not come out opposite (the same goal for both, or a team with no dot tracked), nobody can
    say whose goalkeeper is whose, and the rule writes nothing.
    """
    ball, pitch = tracking.ball, tracking.pitch
    scale = np.array(pitch)
    held = np.full(tracking.n_frames, NOBODY, dtype=np.int8)
    out = wholly_out(ball, pitch)
    k = max(2, int(round(MOVING_S * fps)))
    cap = int(round((HOLD_S + FLIGHT_S) * fps))
    for _, first, stop in _blocks(tracking):
        goals = {team: own_goal_x(tracking, team, first, stop) for team in (HOME, AWAY)}
        if None in goals.values() or goals[HOME] == goals[AWAY]:
            continue
        for a, b in ball_gaps(ball, first, stop):
            last = a - 1
            if b - a > cap or b + k >= stop or out[last] or out[b:b + k + 1].any() or \
                    not np.isfinite(ball[b:b + k + 1, 0]).all():
                continue
            for team, goal in sorted(goals.items()):
                if not in_own_box(ball[last], goal, pitch):
                    continue
                gk = keeper(tracking, last, team, goal)
                if gk is None or np.hypot(*((ball[last] - gk) * scale)) > KEEPER_REACH_M:
                    continue
                now = keeper(tracking, b, team, goal)
                now = gk if now is None else now
                away = (np.hypot(*((ball[b + k] - now) * scale)) -
                        np.hypot(*((ball[b] - now) * scale))) * fps / k
                if away >= MOVING_M_S:
                    held[a:b] = team
    return held


def kept(tracking, fps=FPS):
    """Rules B, B2 and a ball seen off the pitch (C): the D from the clock and the ball's place."""
    return before_kickoff(tracking) | outside_the_halves(tracking) | off_the_pitch(tracking, fps)


def dead_rules(tracking, tags=None, fps=FPS):
    """[(name, mask)] of every dead-ball rule, in a fixed order: B, B2, C (C2 included), F3, and
    with the Smart Tagging D (D2) and D'. tags=None (the practice games) runs the tracking-only
    rules. The lost-ball corrections (F1, F2) never write over any of these."""
    rules = [("before kickoff", before_kickoff(tracking)),
             ("outside the halves", outside_the_halves(tracking)),
             ("ball off the pitch", off_the_pitch(tracking, fps)),
             ("huddle", huddle(tracking, fps))]
    if tags is not None:
        rules.append(("restarts from the tags", restarts(tracking, tags, fps)))
    return rules


def dead_mask(tracking, tags=None, fps=FPS):
    """Every frame a dead-ball rule calls dead (dead_rules, together)."""
    dead = np.zeros(tracking.n_frames, dtype=bool)
    for _, mask in dead_rules(tracking, tags, fps):
        dead |= mask
    return dead
