import numpy as np
import pytest

import dead_ball
from tracking import Tracking

PITCH = (100.0, 50.0)          # metres; round numbers keep the expected frames exact
INSIDE = (0.5, 0.5)
NAN = (np.nan, np.nan)


RUNNERS = ((0.2, 0.1), (0.8, 0.9))       # one home, one away, far (> 15 m) from every ball here


def tracking_of(ball, blocks=None, kickoffs=None, runners_from=None):
    """A Tracking with one home player on the centre spot and the given ball path. With
    runners_from, two more players (one per team, at RUNNERS) stand still and then run along x at
    3 m/s from that frame on: the players' sign of a restart (the restart test's (b))."""
    ball = np.array(ball, dtype=float)
    players = np.full((len(ball), 1, 2), 0.5)
    team = np.array([0], dtype=np.int8)
    if runners_from is not None:
        players = np.concatenate((players, runners(len(ball), runners_from)), axis=1)
        team = np.array([0, 0, 1], dtype=np.int8)
    return Tracking(ball, players, team, PITCH, blocks or [], kickoffs)


def runners(n, start, spots=RUNNERS, speed=3.0):
    """(n, len(spots), 2): players standing at `spots` until `start`, then running along x."""
    out = np.tile(np.array(spots, dtype=float), (n, 1, 1))
    ran = np.clip(np.arange(n) - start, 0, None) * speed / 25 / PITCH[0]
    out[:, :, 0] += ran[:, None]
    return out


def restart_scene(ball):
    """tracking_of(ball) with the runners starting when its last 30-frame kick does (every
    restart fixture below ends with moving(30, ...): the kick is from len(ball) - 31)."""
    return tracking_of(ball, runners_from=len(ball) - 31)


def path(*pieces):
    """A ball path from (n_frames, (x, y)) pieces."""
    return [xy for count, xy in pieces for _ in range(count)]


def frames(mask):
    """(first, stop) of the single True run in mask, or None if there is none."""
    hits = np.flatnonzero(mask)
    if len(hits) == 0:
        return None
    assert np.all(np.diff(hits) == 1), "more than one stretch"
    return int(hits[0]), int(hits[-1]) + 1


# -- rule B: before kickoff --------------------------------------------------------------------

@pytest.mark.parametrize("kickoffs, expected", [
    ({1: 10, 2: 65}, [(60, 65)]),         # half 1 kicks off on its first graded frame
    ({1: 15, 2: 60}, [(10, 15)]),
    ({1: 12, 2: 70}, [(10, 12), (60, 70)]),
    (None, []),                           # practice games: no kickoffs known, nothing marked
])
def test_before_kickoff_marks_graded_frames_before_each_kickoff(kickoffs, expected):
    tracking = tracking_of(path((100, INSIDE)), [(1, 10, 50), (2, 60, 100)], kickoffs)
    want = np.zeros(100, dtype=bool)
    for first, stop in expected:
        want[first:stop] = True
    np.testing.assert_array_equal(dead_ball.before_kickoff(tracking), want)


# -- rule B2: outside the halves ---------------------------------------------------------------

@pytest.mark.parametrize("blocks, expected", [
    ([(1, 10, 50), (2, 60, 90)], [(0, 10), (50, 60), (90, 100)]),   # before, halftime, after
    ([(1, 0, 50), (2, 50, 100)], []),     # the halves cover every frame
    ([], []),                             # no blocks: one period over all frames
])
def test_outside_the_halves_is_dead_and_inside_untouched(blocks, expected):
    tracking = tracking_of(path((100, INSIDE)), blocks)
    want = np.zeros(100, dtype=bool)
    for first, stop in expected:
        want[first:stop] = True
    np.testing.assert_array_equal(dead_ball.outside_the_halves(tracking), want)


def test_dead_mask_writes_outside_the_halves():
    tracking = tracking_of(path((100, INSIDE)), [(1, 10, 50), (2, 60, 90)])
    assert np.flatnonzero(dead_ball.dead_mask(tracking)).tolist() == \
        list(range(10)) + list(range(50, 60)) + list(range(90, 100))


# -- rule C: the whole ball over a line --------------------------------------------------------

@pytest.mark.parametrize("xy, out", [
    ((-0.11 / 100, 0.5), True),           # exactly one radius past the goal line at x = 0
    ((1 + 0.11 / 100, 0.5), True),        # ... at x = length
    ((0.5, -0.11 / 50), True),            # ... past the touchline at y = 0
    ((0.5, 1 + 0.11 / 50), True),         # ... at y = width
    ((-0.10 / 100, 0.5), False),          # 10 cm past: part of the ball is still on the line
    ((0.5, 1 + 0.10 / 50), False),
    ((0.0, 0.5), False),                  # centre on the line
    ((0.5, 0.5), False),
    ((np.nan, np.nan), False),            # not tracked: no evidence either way
])
def test_wholly_out_needs_the_whole_ball_over_the_line(xy, out):
    assert dead_ball.wholly_out(np.array([xy]), PITCH)[0] == out


OUT = (-0.005, 0.5)                       # 50 cm past the goal line

def test_off_the_pitch_runs_from_the_first_frame_out_to_the_frame_before_back():
    tracking = tracking_of(path((20, INSIDE), (40, OUT), (40, INSIDE)))
    assert frames(dead_ball.off_the_pitch(tracking)) == (20, 60)


@pytest.mark.parametrize("n_out, expected", [
    (2, None),                            # a two-frame blip over the line is jitter
    (24, None),                           # back inside within 1 s: never out
    (25, (20, 45)),                       # 1 s: a stoppage
])
def test_off_for_under_a_second_writes_nothing(n_out, expected):
    tracking = tracking_of(path((20, INSIDE), (n_out, OUT), (60, INSIDE)))
    assert frames(dead_ball.off_the_pitch(tracking)) == expected


def test_a_short_blip_back_inside_joins_two_out_stretches():
    tracking = tracking_of(path((20, INSIDE), (20, OUT), (3, INSIDE), (27, OUT), (30, INSIDE)))
    assert frames(dead_ball.off_the_pitch(tracking)) == (20, 70)


def test_out_until_the_end_of_the_period_stays_inside_its_period():
    tracking = tracking_of(path((60, INSIDE), (60, OUT), (80, INSIDE)),
                           blocks=[(1, 0, 100), (2, 100, 200)])
    dead = dead_ball.off_the_pitch(tracking)
    assert frames(dead[:100]) == (60, 100)     # out until the end of half 1
    assert frames(dead[100:]) is None          # half 2 opens with 20 frames out: under 1 s


def test_out_while_untracked_continues_the_stoppage():
    tracking = tracking_of(path((20, INSIDE), (10, OUT), (50, NAN), (5, OUT), (15, INSIDE)))
    assert frames(dead_ball.off_the_pitch(tracking)) == (20, 85)


# -- rule C2: the ball crossed the line while the tracker had lost it --------------------------

def towards_the_goal_line(n, start_x, step_x):
    """n tracked frames moving along x by step_x per frame, mid-width."""
    return [(start_x + step_x * k, 0.5) for k in range(n)]


def test_crossing_in_a_gap_is_extrapolated_from_the_last_velocity():
    # 0.2 m per frame towards x = 0; last seen 1.2 m inside on frame 29. Wholly out at
    # x <= -0.11 m: (1.2 + 0.11) / 0.2 = 6.55 -> 7 frames on, frame 36.
    ball = towards_the_goal_line(30, 0.07, -0.002) + path((20, NAN), (30, OUT), (20, INSIDE))
    assert frames(dead_ball.off_the_pitch(tracking_of(ball))) == (36, 80)


@pytest.mark.parametrize("approach", [
    "standing still",                     # no velocity
    "too slow",                           # the estimate comes after the ball was seen out
    "one tracked frame",                  # no second frame to measure a velocity
])
def test_crossing_falls_back_to_the_gap_start_when_it_cannot_be_estimated(approach):
    if approach == "standing still":
        before = path((30, (0.012, 0.5)))
    elif approach == "too slow":
        before = towards_the_goal_line(30, 0.07, -0.00001)
    else:
        before = path((28, NAN), (1, NAN), (1, (0.012, 0.5)))
        before[27] = INSIDE                # an older sighting, not adjacent to the last one
    ball = before + path((20, NAN), (30, OUT), (20, INSIDE))
    assert frames(dead_ball.off_the_pitch(tracking_of(ball))) == (30, 80)


def test_crossing_frame_is_none_without_two_tracked_frames():
    ball = np.array(path((1, NAN), (1, INSIDE)))
    assert dead_ball.crossing_frame(ball, 1, PITCH) is None


# -- the rules together ------------------------------------------------------------------------

def test_dead_mask_is_both_rules_and_no_more():
    tracking = tracking_of(path((40, INSIDE), (30, OUT), (30, INSIDE)), [(1, 0, 100)], {1: 10})
    mask = dead_ball.dead_mask(tracking)
    assert np.flatnonzero(mask).tolist() == list(range(10)) + list(range(40, 70))


# -- rule D: restarts from the Smart Tagging ---------------------------------------------------

KICK = (0.004, 0.0)                       # 0.4 m per frame along x: 10 m/s at 25 fps
NET = (1.005, 0.5)                        # 50 cm past the goal line at x = length


def moving(n, start, step=KICK):
    """n tracked frames after `start`, the ball moving by `step` per frame. Put it right after
    a frame at `start`: that frame is then the one the ball is kicked from."""
    return [(start[0] + step[0] * (k + 1), start[1] + step[1] * (k + 1)) for k in range(n)]


def tag(code, start, end, kind=None):
    """A Smart Tagging instance as the starter loader returns it."""
    return {"code": code, "start_frame": start, "end_frame": end,
            "tags": {"Type": kind} if kind else {}}


@pytest.mark.parametrize("speed, kicked", [
    (0.0, False),                         # standing still
    (2.0, False),                         # rolled to the taker
    (3.0, False),                         # carried by a jogging player
    (5.0, True),                          # a soft restart pass
    (10.0, True),
])
def test_a_kick_is_the_ball_moving_at_4_m_s_or_more(speed, kicked):
    step = (speed / 25 / 100, 0.0)
    ball = np.array(path((31, (0.3, 0.5))) + moving(20, (0.3, 0.5), step))
    hits = np.flatnonzero(dead_ball.kicked(ball, PITCH))
    assert (hits[0] if len(hits) else None) == (30 if kicked else None)


def test_a_one_frame_tracking_jump_is_not_a_kick():
    ball = path((30, INSIDE), (1, (0.52, 0.5)), (30, INSIDE))   # 2 m away for one frame
    assert not dead_ball.kicked(np.array(ball), PITCH).any()


def lost_then_kicked():
    """Live play, the ball lost on 40-89, back inside standing still, kicked from frame 120."""
    return moving(40, (0.3, 0.5)) + path((50, NAN), (31, INSIDE)) + moving(30, INSIDE)


def out_then_placed(kick_from=110):
    """Live play, the ball tracked over the goal line on 30-59, placed in the goal area, kicked."""
    spot = (0.05, 0.5)
    return (moving(30, (0.3, 0.5)) + path((30, OUT), (kick_from - 59, spot)) +
            moving(30, spot))


def gap_then_seen_out():
    """Towards the goal line at 5 m/s, lost on 30-49, seen out, placed, kicked from frame 100."""
    spot = (0.05, 0.5)
    return (towards_the_goal_line(30, 0.07, -0.002) + path((20, NAN), (30, OUT), (21, spot)) +
            moving(30, spot))


@pytest.mark.parametrize("ball, set_piece, expected", [
    # lost before the tag, back after it: dead from the loss to the kick (a throw-in, a corner)
    (lost_then_kicked(), tag("SET PIECES", 60, 200, "Throw-in"), (40, 120)),
    # tracked throughout (most goal kicks): dead from the ball out to the kick, not to its return
    (out_then_placed(), tag("SET PIECES", 80, 200, "Goal Kick"), (30, 110)),
    # lost, then seen already off the pitch: the stoppage starts at rule C2's estimated crossing
    (gap_then_seen_out(), tag("SET PIECES", 85, 200, "Goal Kick"), (36, 100)),
    # the first kick inside the tag (100) fails the restart test and no kick passes it within
    # 1.6 s: the first kick stays the restart, though the players react to a kick 3.6 s later
    # (a review fix: a missed restart never stretches D into the live play after it)
    (moving(40, (0.3, 0.5)) + path((50, NAN), (11, INSIDE)) + moving(10, INSIDE) +
     path((80, (0.54, 0.5))) + moving(30, (0.54, 0.5)), tag("SET PIECES", 60, 300, "Throw-in"),
     (40, 100)),
    # the ball never lost before the restart: a tag is evidence, not an answer
    (path((80, INSIDE)) + moving(30, INSIDE), tag("SET PIECES", 40, 200, "Free Kick"), None),
    # no kick inside the tag: no restart found, nothing written
    (lost_then_kicked(), tag("SET PIECES", 60, 100, "Throw-in"), None),
    # the tag starts after the restart, the ball already moving: nothing written
    (out_then_placed(), tag("SET PIECES", 115, 200, "Goal Kick"), None),
    # lost for less than 1 s before the kick: under rule C's floor, nothing written
    (moving(40, (0.3, 0.5)) + path((10, NAN), (6, INSIDE)) + moving(30, INSIDE),
     tag("SET PIECES", 45, 200, "Free Kick"), None),
    # the first kick-off of a half is rule B's; other codes are not restarts
    (lost_then_kicked(), tag("SET PIECES", 60, 200, "Kick Off (Start)"), None),
    (lost_then_kicked(), tag("BUILD UP", 60, 200), None),
])
def test_a_set_piece_is_dead_from_the_ball_lost_to_the_frame_before_the_kick(ball, set_piece,
                                                                               expected):
    assert frames(dead_ball.restarts(restart_scene(ball), [set_piece])) == expected


def test_goal_kick_tracked_throughout_reaches_further_than_rule_c():
    tracking = tracking_of(out_then_placed())
    assert frames(dead_ball.off_the_pitch(tracking)) == (30, 60)   # control: rule C alone


def scored_then_kick_off():
    """In the net on 30-59, kicked about inside on 60-99, on the centre spot, kicked from 200."""
    return (moving(30, (0.8, 0.5)) + path((30, NET)) + moving(40, (0.6, 0.5)) +
            path((101, INSIDE)) + moving(30, INSIDE))


def test_a_goal_is_dead_from_the_ball_in_the_net_to_the_kick_off():
    tags = [tag("GOALS", 20, 40), tag("SET PIECES", 150, 260, "Kick Off (After Goal)")]
    assert frames(dead_ball.restarts(restart_scene(scored_then_kick_off()), tags)) == (30, 200)


def test_a_goal_without_a_kick_off_after_it_writes_nothing():
    tags = [tag("GOALS", 20, 40)]
    assert frames(dead_ball.restarts(restart_scene(scored_then_kick_off()), tags)) is None


def test_the_goal_is_the_first_time_over_a_goal_line_after_the_last_other_restart():
    corner = (0.99, 0.01)
    ball = (moving(10, (0.8, 0.5)) + path((30, NET), (21, corner)) +       # out for a corner
            moving(40, corner, (-0.004, 0.0)) + path((30, NET), (100, INSIDE)) +   # scored
            moving(30, INSIDE))                                            # kicked from 230
    tags = [tag("GOALS", 0, 120), tag("SET PIECES", 45, 90, "Corner Kick"),
            tag("SET PIECES", 180, 300, "Kick Off (After Goal)")]
    tracking = tracking_of(ball)
    kicks = dead_ball.kicked(tracking.ball, PITCH)
    assert dead_ball.goal_stoppage(tracking, kicks, tags[0], tags) == (101, 230)


# -- rule D', its fallback: a kick-off tag with no kick inside it ------------------------------

KICK_OFF_TAG = ("SET PIECES", 150, 260, "Kick Off (After Goal)")           # the kick (200) inside
LATE_TAG = ("SET PIECES", 150, 190, "Kick Off (After Goal)")               # no kick inside it


def lost_in_the_goal():
    """scored_then_kick_off() with the ball lost on 30-59 instead of seen in the net: kicked
    towards the goal up to 29, lost, kicked about inside on 60-99, on the centre spot, kicked
    from 200."""
    return (moving(30, (0.8, 0.5)) + path((30, NAN)) + moving(40, (0.6, 0.5)) +
            path((101, INSIDE)) + moving(30, INSIDE))


def goal_frames(monkeypatch, ball, kick_off, tag_restart):
    """frames() of the tag rules for a GOALS tag on 20-39 and the kick-off tag, with the switch
    set as a module Param (read when called), through the pipeline's own path."""
    monkeypatch.setattr(dead_ball, "GOAL_TAG_RESTART", tag_restart)
    return frames(dead_ball.restarts(restart_scene(ball), [tag("GOALS", 20, 40), tag(*kick_off)]))


@pytest.mark.parametrize("tag_restart", [False, True])
def test_a_goal_seen_over_the_line_with_a_kick_in_the_tag_is_as_before(monkeypatch, tag_restart):
    assert goal_frames(monkeypatch, scored_then_kick_off(), KICK_OFF_TAG, tag_restart) == (30, 200)


@pytest.mark.parametrize("kick_off, tag_restart", [
    (KICK_OFF_TAG, False), (KICK_OFF_TAG, True), (LATE_TAG, False), (LATE_TAG, True)])
def test_a_goal_whose_ball_is_never_seen_over_the_line_writes_nothing(monkeypatch, kick_off,
                                                                      tag_restart):
    # the ball lost as it goes in: no fallback starts the stoppage at the loss
    assert goal_frames(monkeypatch, lost_in_the_goal(), kick_off, tag_restart) is None


@pytest.mark.parametrize("tag_restart, expected", [
    (False, None),                        # no kick inside the kick-off tag: nothing, as before
    (True, (30, 150)),                    # dead from the ball in the net to the tag's start
])
def test_a_goal_with_no_kick_inside_the_kick_off_tag_is_dead_to_the_tags_start(
        monkeypatch, tag_restart, expected):
    assert goal_frames(monkeypatch, scored_then_kick_off(), LATE_TAG, tag_restart) == expected


def test_the_goal_fallback_keeps_the_one_second_floor(monkeypatch):
    # in the net from 140, the late tag starts on 150: 0.4 s, under the floor
    ball = path((140, INSIDE), (30, NET), (60, INSIDE))
    monkeypatch.setattr(dead_ball, "GOAL_TAG_RESTART", True)
    tags = [tag("GOALS", 130, 150), tag(*LATE_TAG)]
    assert frames(dead_ball.restarts(tracking_of(ball), tags)) is None
    tags = [tag("GOALS", 130, 150), tag("SET PIECES", 165, 190, "Kick Off (After Goal)")]
    assert frames(dead_ball.restarts(tracking_of(ball), tags)) == (140, 165)   # 1 s: written


def test_the_goal_switch_defaults_to_the_param_and_an_argument_beats_it(monkeypatch):
    tracking = restart_scene(scored_then_kick_off())
    tags = [tag("GOALS", 20, 40), tag(*LATE_TAG)]
    kicks = dead_ball.kicked(tracking.ball, PITCH)
    monkeypatch.setattr(dead_ball, "GOAL_TAG_RESTART", False)
    assert dead_ball.goal_stoppage(tracking, kicks, tags[0], tags) is None
    assert dead_ball.goal_stoppage(tracking, kicks, tags[0], tags, tag_restart=True) == (30, 150)
    monkeypatch.setattr(dead_ball, "GOAL_TAG_RESTART", True)
    assert dead_ball.goal_stoppage(tracking, kicks, tags[0], tags) == (30, 150)
    assert dead_ball.goal_stoppage(tracking, kicks, tags[0], tags, tag_restart=False) is None


def test_rule_d_and_its_shared_helper_give_the_same_start():
    # the helper both rules share: rule D's stoppage is its start, floor applied, nothing else
    tracking = restart_scene(lost_then_kicked())
    kicks = dead_ball.kicked(tracking.ball, PITCH)
    throw_in = tag("SET PIECES", 60, 200, "Throw-in")
    assert dead_ball.set_piece_stoppage(tracking, kicks, throw_in) == (40, 120)
    assert dead_ball.lost_ball_start(tracking, kicks, (0, tracking.n_frames), 60, 120) == 40
    never_lost = tracking_of(path((80, INSIDE)) + moving(30, INSIDE))
    kicks = dead_ball.kicked(never_lost.ball, PITCH)
    assert dead_ball.lost_ball_start(never_lost, kicks, (0, never_lost.n_frames), 40, 80) is None


def test_dead_mask_runs_the_tag_rules_only_with_tags():
    tracking = restart_scene(lost_then_kicked())
    assert frames(dead_ball.dead_mask(tracking)) is None              # practice path: no tags
    assert frames(dead_ball.dead_mask(tracking, [tag("SET PIECES", 60, 200, "Throw-in")])) == \
        (40, 120)


def test_every_set_piece_tag_is_timed_alike_whatever_its_type():
    tracking = restart_scene(lost_then_kicked())
    for kind in ("Throw-in", "Corner Kick", "Goal Kick", "Free Kick", None):
        assert frames(dead_ball.restarts(tracking, [tag("SET PIECES", 60, 200, kind)])) == \
            (40, 120), kind


# -- the restart test: the ball kicked and the players show it was played ---------------------

def kick_scene(players, start=60, n=120, kick_at=(0.5, 0.5), still=None):
    """The ball still at `kick_at` and kicked from frame `start` at 10 m/s; `players` is a list of
    (team, spot, speed m/s, from frame, direction (dx, dy)) for players who stand at `spot` and
    move from that frame on; `still` adds players who never move."""
    ball = np.array(path((start + 1, kick_at)) + moving(n - start - 1, kick_at), dtype=float)
    spots, team = [], []
    track = np.empty((n, len(players), 2))
    for k, (side, spot, speed, frame, direction) in enumerate(players):
        step = np.clip(np.arange(n) - frame, 0, None) * speed / 25
        track[:, k, 0] = spot[0] + step * direction[0] / PITCH[0]
        track[:, k, 1] = spot[1] + step * direction[1] / PITCH[1]
        team.append(side)
    return Tracking(ball, track, np.array(team, dtype=np.int8), PITCH, [], None)


RIGHT, LEFT = (1.0, 0.0), (-1.0, 0.0)


@pytest.mark.parametrize("players, expected", [
    # (b) a player of each team, far from the ball, speeds up as it is kicked: the restart
    ([(0, (0.2, 0.1), 3.0, 60, RIGHT), (1, (0.8, 0.9), 3.0, 60, LEFT)], 60),
    # a fast ball and nobody moves (a pass to the taker, a ball boy's throw): not the restart
    ([(0, (0.2, 0.1), 0.0, 60, RIGHT), (1, (0.8, 0.9), 0.0, 60, LEFT)], None),
    # far players of one team only: not enough
    ([(0, (0.2, 0.1), 3.0, 60, RIGHT), (0, (0.8, 0.9), 3.0, 60, LEFT)], None),
    # (a) a player 8 m from the ball speeds up and closes on it (the ball goes along +x)
    ([(0, (0.5, 0.34), 3.0, 60, (1.0, 1.0))], 60),
    # ... the same player running away from the ball: not chasing it
    ([(0, (0.5, 0.34), 3.0, 60, (-1.0, -1.0))], None),
    # far players of both teams barely speed up (0.4 m/s, under the 0.5 m/s gain)
    ([(0, (0.2, 0.1), 0.4, 60, RIGHT), (1, (0.8, 0.9), 0.4, 60, LEFT)], None),
    # far players of both teams already running before the kick, no faster after: no sign
    ([(0, (0.2, 0.1), 3.0, 0, RIGHT), (1, (0.8, 0.9), 3.0, 0, LEFT)], None),
])
def test_a_restart_is_a_kick_the_players_react_to(players, expected):
    scene = kick_scene(players)
    kicks = dead_ball.kicked(scene.ball, PITCH)
    assert kicks[60]                                            # control: the ball is kicked
    hits = np.flatnonzero(dead_ball.played(scene, kicks))
    assert (int(hits[0]) if len(hits) else None) == expected


def test_a_player_off_the_pitch_is_not_counted():
    # the away player warms up by the touchline: only the home player is on the pitch
    scene = kick_scene([(0, (0.2, 0.1), 3.0, 60, RIGHT), (1, (0.8, 1.05), 3.0, 60, LEFT)])
    assert not dead_ball.played(scene, dead_ball.kicked(scene.ball, PITCH)).any()


# -- rule D2: a stoppage starts where the players slow down ------------------------------------

@pytest.mark.parametrize("signal, expected", [
    ([2.0] * 60 + [1.0] * 140, 35),       # running at the loss, slower from 60: 60 less 1 s
    ([2.0] * 30 + [1.0] * 170, 20),       # slower within a second of the loss: the loss
    ([1.0] * 200, 20),                    # already slow at the loss: the loss (level 4: the tag)
    ([1.5] * 200, 20),                    # a bit clear only (under 1.54): the loss
    ([2.0] * 200, 20),                    # running all the way to the restart: the loss
    ([np.nan] * 200, 20),                 # players unknown: the loss
])
def test_a_lost_ball_stoppage_starts_where_the_players_slow_down(signal, expected):
    assert dead_ball.slowdown_start(np.array(signal), 20, 150) == expected


def slowdown_scene(stop_at):
    """Live play: everyone runs along x at 3 m/s until `stop_at`, then stands; the ball is kicked
    about until frame 40, lost on 40-149, placed at the centre on 150-219 and kicked from 219,
    when one player of each team starts running again."""
    n = 250
    ball = np.array(moving(40, (0.3, 0.5)) + path((110, NAN), (70, INSIDE)) + moving(30, INSIDE))
    spots = [(0.2, 0.1), (0.3, 0.2), (0.8, 0.9), (0.7, 0.8)]
    players = np.tile(np.array(spots), (n, 1, 1))
    players[:, :, 0] += (np.clip(np.arange(n), None, stop_at) * 3.0 / 25 / PITCH[0])[:, None]
    players[:, [0, 2], 0] += (np.clip(np.arange(n) - 219, 0, None) * 3.0 / 25 / PITCH[0])[:, None]
    return Tracking(ball, players, np.array([0, 0, 1, 1], dtype=np.int8), PITCH, [], None)


@pytest.mark.parametrize("stop_at, expected", [
    (100, (74, 219)),     # running when the ball is lost: from where they slow (99) less 1 s
    (0, (40, 219)),       # standing when the ball is lost: from the loss
])
def test_rule_d_starts_a_lost_ball_stoppage_where_the_players_slow_down(stop_at, expected):
    tracking = slowdown_scene(stop_at)
    assert frames(dead_ball.restarts(tracking, [tag("SET PIECES", 160, 300, "Free Kick")])) == \
        expected


# -- fixes from a review of the dead-ball rules -------------------------------------------------

def marks(n, at):
    out = np.zeros(n, dtype=bool)
    out[list(at)] = True
    return out


@pytest.mark.parametrize("kicks, ready, end, expected", [
    ([10], [10], 100, 10),                # the first kick passes the test: the restart
    ([10], [], 100, 10),                  # fails, nothing passes after it: still the restart
    ([10, 40], [40], 100, 40),            # fails, a kick 1.2 s later passes: that one
    ([10, 50], [50], 100, 50),            # ... exactly 1.6 s later: still that one
    ([10, 51], [51], 100, 10),            # ... 1.64 s later: too late, the first kick stays
    ([10, 40], [40], 30, 10),             # the passing kick is after the tag's end
    ([10, 40], None, 100, 10),            # no restart test given: the first kick
    ([], [], 100, None),                  # no kick inside the tag
])
def test_the_first_kick_is_the_restart_unless_a_passing_kick_comes_soon(kicks, ready, end,
                                                                         expected):
    ready = None if ready is None else marks(120, ready)
    assert dead_ball.restart_frame(marks(120, kicks), 5, end, ready) == expected


def test_a_fake_kick_followed_quickly_by_a_kick_the_players_react_to_gives_that_kick():
    # a pass to the taker from 100 nobody reacts to; the real kick from 129, 1.16 s later, when
    # far players of both teams start running (at 0.7 m/s: the pass's next 2 s see too little)
    ball = (moving(40, (0.3, 0.5)) + path((50, NAN), (11, INSIDE)) + moving(10, INSIDE) +
            path((19, (0.54, 0.5))) + moving(30, (0.54, 0.5)))
    players = np.concatenate((np.full((len(ball), 1, 2), 0.5),
                              runners(len(ball), 129, speed=0.7)), axis=1)
    tracking = Tracking(np.array(ball), players, np.array([0, 0, 1], dtype=np.int8), PITCH, [],
                        None)
    kicks = dead_ball.kicked(tracking.ball, PITCH)
    ready = dead_ball.played(tracking, kicks)
    assert kicks[100] and not ready[100:129].any() and ready[129]   # control: fake, then pass
    assert frames(dead_ball.restarts(tracking, [tag("SET PIECES", 60, 300, "Throw-in")])) == \
        (40, 129)


def test_the_slower_team_is_unknown_when_either_team_is_unknown():
    # home runs at 3 m/s throughout; away stands, and is off the pitch on frames 100-199
    n = 300
    players = np.concatenate((runners(n, 0, ((0.1, 0.2), (0.1, 0.3))),
                              np.tile(np.array([(0.8, 0.5), (0.7, 0.5)]), (n, 1, 1))), axis=1)
    players[100:200, 2:, 1] = 1.1
    tracking = Tracking(np.full((n, 2), 0.5), players, np.array([0, 0, 1, 1], dtype=np.int8),
                        PITCH, [], None)
    signal = dead_ball.slower_team_speed(tracking)
    # control: both known, away still (frames 45-99 see the away dots' jump off the pitch)
    assert np.nanmax(signal[:40]) < 0.1
    assert np.isnan(signal[150])                                  # one team alone says nothing


@pytest.mark.parametrize("signal, expected", [
    ([2.0] * 40 + [np.nan] * 20 + [1.0] * 140, 35),   # unknown 40-59, slower from 60: 60 less 1 s
    ([2.0] * 40 + [np.nan] * 160, 20),                # never seen slowing down: the loss
])
def test_unknown_players_are_never_read_as_slowed_down(signal, expected):
    assert dead_ball.slowdown_start(np.array(signal), 20, 150) == expected


def edge_scene(blocks, run_from, speed=6.0, n=300):
    """The ball still on the centre spot, kicked from frame 110 at 10 m/s; one far player of each
    team stands and then runs along x at `speed` from `run_from`."""
    ball = np.array(path((111, INSIDE)) + moving(n - 111, INSIDE), dtype=float)
    players = runners(n, run_from, speed=speed)
    return Tracking(ball, players, np.array([0, 1], dtype=np.int8), PITCH, blocks, None)


def test_the_restart_test_never_reads_the_players_across_a_halfs_edge():
    # half 1 ends at 150; the players start running in half 2, 1.6 s after the kick
    one_period = edge_scene([], 150)
    kicks = dead_ball.kicked(one_period.ball, PITCH)
    assert kicks[110] and dead_ball.played(one_period, kicks)[110]   # control: no edge, a sign
    halves = edge_scene([(1, 0, 150), (2, 150, 300)], 150)
    assert not dead_ball.played(halves, kicks)[110]


def test_the_slower_team_never_reads_across_a_halfs_edge():
    # both teams run at 6 m/s in half 1 (to frame 100) and stand in half 2
    n = 250
    players = runners(n, 0, ((0.1, 0.2), (0.1, 0.8)), speed=6.0)
    players[100:] = players[99]
    tracking = Tracking(np.full((n, 2), 0.5), players, np.array([0, 1], dtype=np.int8), PITCH,
                        [(1, 0, 100), (2, 100, 250)], None)
    signal = dead_ball.slower_team_speed(tracking)
    assert signal[60] > dead_ball.LIVE_BAR_M_S                    # control: running in half 1
    assert np.nanmax(signal[100:]) == 0.0                         # standing, from the edge on


# -- rule F1: a ball in flight -----------------------------------------------------------------

def kicked_then_lost(gap, back=INSIDE, step=KICK, flying=10):
    """Still on 0-19, kicked from frame 19, tracked in flight for `flying` frames, lost for
    `gap` frames from frame 20 + flying, then seen at `back` for 30 frames."""
    return path((20, INSIDE)) + moving(flying, INSIDE, step) + path((gap, NAN), (30, back))


@pytest.mark.parametrize("ball, expected", [
    (kicked_then_lost(50), (30, 80)),               # back inside 2 s later: in flight, live
    (kicked_then_lost(100), (30, 130)),             # 4 s: the longest a kicked ball stays up
    (kicked_then_lost(101), None),                  # longer: left as it is
    (kicked_then_lost(50, back=OUT), None),         # next seen off the pitch: rule C2 decides
    (kicked_then_lost(50, step=(0.0012, 0.0)), None),   # 3 m/s: rolled, not kicked
    (kicked_then_lost(50, flying=40), None),        # lost 1.6 s after the kick: not right after
])
def test_a_ball_lost_right_after_a_kick_and_back_inside_soon_was_in_flight(ball, expected):
    assert frames(dead_ball.in_flight(tracking_of(ball))) == expected


# -- rule F2: the goalkeeper holding the ball --------------------------------------------------

KEEPER = (0.04, 0.5)                      # 4 m off the home goal line at x = 0, mid-goal


AWAY_SPOTS = ((0.6, 0.4), (0.7, 0.6))     # the away players stand in their own half (x > 0.5)


def keeper_scene(last=KEEPER, keeper_at=KEEPER, gap=100, release=KICK, back=None,
                 away=AWAY_SPOTS):
    """Home defends x = 0 (keeper + 1 outfield player), away attacks (2 players, at `away`). A
    shot comes in at 10 m/s and is last seen at `last`, lost for `gap` frames, then seen at `back`
    (the keeper's spot) moving by `release` per frame (his kick) for 30 frames."""
    back = keeper_at if back is None else back
    shot = [(last[0] + 0.004 * (20 - k), last[1]) for k in range(1, 21)]
    ball = path((20, (last[0] + 0.08, last[1]))) + shot + path((gap, NAN))
    ball += [(back[0] + release[0] * k, back[1] + release[1] * k) for k in range(30)]
    n = len(ball)
    players = np.tile(np.array([keeper_at, (0.3, 0.5)] + list(away)), (n, 1, 1))
    return Tracking(np.array(ball, dtype=float), players, np.array([0, 0, 1, 1], dtype=np.int8),
                    PITCH, [], None)


@pytest.mark.parametrize("scene, expected", [
    (keeper_scene(), (40, 140)),                                # caught, held 4 s, kicked away
    (keeper_scene(gap=442), (40, 482)),                 # 17.7 s: 13.7 of holding + 4 of flight
    (keeper_scene(gap=443), None),                      # longer than 9 catches in 10 are held
    (keeper_scene(last=(0.04, 0.54)), (40, 140)),               # 2 m from him: within reach
    (keeper_scene(last=(0.04, 0.62)), None),                    # 6 m from him: not his ball
    (keeper_scene(last=(0.2, 0.5), keeper_at=(0.2, 0.5)), None),   # 20 m out: outside his box
    (keeper_scene(release=(0.0, 0.0)), None),                   # seen still: placed, not released
    (keeper_scene(back=(-0.005, 0.5)), None),                   # seen off the pitch: rule C
])
def test_a_ball_lost_with_the_goalkeeper_is_his_team_until_he_releases_it(scene, expected):
    held = dead_ball.keeper_holding(scene)
    assert frames(held >= 0) == expected
    assert set(held[held >= 0].tolist()) <= {0}                 # the home keeper's team


@pytest.mark.parametrize("away, expected", [
    (AWAY_SPOTS, (40, 140)),                    # control: opposite goals, the home keeper's ball
    (((0.35, 0.4), (0.45, 0.6)), None),         # away camped in home's half: both given x = 0
    ((NAN, NAN), None),                         # no away dot tracked: its goal is unknown
])
def test_the_keeper_rule_writes_nothing_unless_the_teams_goals_are_opposite(away, expected):
    scene = keeper_scene(away=away)
    goals = [dead_ball.own_goal_x(scene, team, 0, scene.n_frames) for team in (0, 1)]
    assert (goals == [0.0, 1.0]) == (expected is not None)               # the fixture is as said
    held = dead_ball.keeper_holding(scene)
    assert frames(held >= 0) == expected
    assert set(held[held >= 0].tolist()) <= {0}


def test_the_keeper_is_his_teams_player_nearest_its_own_goal():
    scene = keeper_scene()
    assert dead_ball.own_goal_x(scene, 0, 0, scene.n_frames) == 0.0
    assert dead_ball.own_goal_x(scene, 1, 0, scene.n_frames) == 1.0
    np.testing.assert_array_equal(dead_ball.keeper(scene, 0, 0, 0.0), KEEPER)


# -- rule F3: a huddle -------------------------------------------------------------------------

def huddle_scene(home=3, away=3, speed=0.0, spot=(0.5, 0.5), lost=(100, 150), n=250):
    """`home` + `away` players within 2 m of `spot` (all moving along x at `speed` m/s), two far
    away players (one each), the ball tracked far off except on frames `lost`."""
    offsets = [(0.01 * (k % 3), 0.02 * (k // 3)) for k in range(home + away)]   # <= 2 m apart
    step = speed / 25 / PITCH[0]
    players = np.empty((n, home + away + 2, 2))
    for k, (dx, dy) in enumerate(offsets):
        players[:, k, 0] = spot[0] + dx + step * np.arange(n)
        players[:, k, 1] = spot[1] + dy
    players[:, -2], players[:, -1] = (0.1, 0.1), (0.9, 0.9)
    team = np.array([0] * home + [1] * away + [0, 1], dtype=np.int8)
    ball = np.tile([0.2, 0.8], (n, 1))
    if lost:
        ball[lost[0]:lost[1]] = np.nan
    return Tracking(ball, players, team, PITCH, [], None)


@pytest.mark.parametrize("scene, expected", [
    (huddle_scene(), (100, 150)),                       # both teams bunched and still: a stoppage
    (huddle_scene(home=4, away=2), (100, 150)),         # two of the smaller team is enough
    (huddle_scene(speed=0.8), (100, 150)),              # walking slowly
    (huddle_scene(speed=3.0), None),                    # moving fast: a corner's box, not a huddle
    (huddle_scene(home=5, away=1), None),               # one team only, near enough
    (huddle_scene(home=3, away=2), None),               # five players: not many
    (huddle_scene(lost=(100, 120)), None),              # under 1 s
    (huddle_scene(lost=None), None),                    # the ball tracked: not this rule
    (huddle_scene(spot=(0.5, 1.03)), None),             # bunched by the touchline: off the pitch
])
def test_a_huddle_of_both_teams_with_the_ball_lost_is_dead(scene, expected):
    assert frames(dead_ball.huddle(scene)) == expected


def test_dead_mask_writes_a_huddle_without_tags():
    assert frames(dead_ball.dead_mask(huddle_scene())) == (100, 150)
