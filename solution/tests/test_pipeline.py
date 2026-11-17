import inspect
import os

import numpy as np
import pytest

import check_seals
import control
import dead_ball
import pipeline
from common import N_FRAMES, pm, read_codes, sha256_of
from conftest import AWAY_CLUB, HOME_CLUB
from starter.baseline import nearest_player_labels, smooth
from tracking import AWAY, HOME, NOBODY, Tracking, load_competition, nearest_team, read_atd

PITCH = (105.3, 68.0)


def text(labels):
    """'H300 A100' run-length text of a label array, for readable assertions."""
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append("{}{}".format(labels[start], i - start))
            start = i
    return " ".join(out)


def random_atd(path, n_frames, n_slots, seed):
    """Random tracking with missing players and ball, and exact ties, in the raw ATD format."""
    rng = np.random.RandomState(seed)
    with open(path, "w") as handle:
        for frame_id in range(1, n_frames + 1):
            xy = np.round(rng.rand(n_slots, 2), 5)
            if frame_id % 7 == 0:
                xy[1] = xy[0]                      # two slots on the same spot: a tie
            groups = ["NaN,NaN,NaN" if rng.rand() < 0.2 else "{:.5f},{:.5f},1.0".format(x, y)
                      for x, y in xy]
            ball = "NaN,NaN,NaN" if rng.rand() < 0.4 else "{:.5f},{:.5f},0.0".format(*rng.rand(2))
            handle.write("{}:{}:{}\n".format(frame_id, ";".join(groups), ball))


def test_nearest_team_equals_the_organisers_baseline(tmp_path):
    raw = str(tmp_path / "atd.txt")
    n_frames, team = 3000, np.array([0, 1, 0, 1, 1, 0], dtype=np.int8)
    random_atd(raw, n_frames, len(team), seed=7)

    theirs = nearest_player_labels(raw, team, n_frames, PITCH)
    ball, players = read_atd(raw, team, n_frames)
    ours = nearest_team(Tracking(ball, players, team, PITCH))

    assert (theirs != -1).sum() > 1000          # positive control: plenty of frames answered
    assert np.array_equal(ours, theirs)


def test_read_atd_window_keeps_only_requested_frames(tmp_path):
    raw = str(tmp_path / "atd.txt")
    team = np.array([0, 1], dtype=np.int8)
    random_atd(raw, 50, len(team), seed=1)
    full_ball, full_players = read_atd(raw, team, 50)
    ball, players = read_atd(raw, team, 50, frames=(10, 20))
    assert np.all(np.isnan(ball[:10])) and np.all(np.isnan(ball[20:]))
    np.testing.assert_array_equal(ball[10:20], full_ball[10:20])
    np.testing.assert_array_equal(players[10:20], full_players[10:20])


def test_label_v0_is_the_smoothed_nearest_team(tmp_path):
    raw = str(tmp_path / "atd.txt")
    team = np.array([0, 1, 1], dtype=np.int8)
    random_atd(raw, 400, len(team), seed=3)
    tracking = Tracking(*read_atd(raw, team, 400), team, PITCH)
    np.testing.assert_array_equal(pipeline.label_v0(tracking), smooth(nearest_team(tracking), 50))
    # random_atd keeps the ball on the pitch and there are no kickoffs: no dead-ball evidence, and
    # the ball is lost on single frames (no long gap for P4), so over v0 itself (both rules of
    # the base off) the only stage that changes anything is the possession rule P1
    nobody = np.zeros(400, dtype=bool)
    _, gap = control.gap_edges(pipeline.label_v0(tracking), tracking, nobody, pipeline.WINDOW)
    assert not gap.any()
    after_p1, changed = control.control(pipeline.label_v0(tracking), tracking, nobody)
    assert changed.any()                                      # control: P1 does fire here
    np.testing.assert_array_equal(pipeline.label(tracking, near_m=None, fast_m_s=None), after_p1)


def beside_the_ball(n_frames, home_m, away_m):
    """The ball on the centre spot; a home dot home_m metres from it along the pitch and an away
    dot away_m metres from it across the pitch (None: that dot is not tracked)."""
    ball = np.tile([0.5, 0.5], (n_frames, 1))
    players = np.full((n_frames, 2, 2), np.nan)
    if home_m is not None:
        players[:, 0] = (0.5 + home_m / PITCH[0], 0.5)
    if away_m is not None:
        players[:, 1] = (0.5, 0.5 + away_m / PITCH[1])
    return Tracking(ball, players, np.array([0, 1], dtype=np.int8), PITCH)


@pytest.mark.parametrize("home_m, away_m, v0_says, near_says", [
    (1.0, None, HOME, HOME),          # a near dot votes for his team
    (2.9, None, HOME, HOME),
    (2.0, 1.0, AWAY, AWAY),           # the nearer of two near dots
    (2.9, 40.0, HOME, HOME),
    (3.1, None, HOME, NOBODY),        # a far dot does not vote
    (40.0, 3.1, AWAY, NOBODY),        # nor does the nearest of two far dots
    (None, None, NOBODY, NOBODY),     # no dot at all
])
def test_the_nearest_dot_votes_only_within_three_metres(home_m, away_m, v0_says, near_says):
    tracking = beside_the_ball(5, home_m, away_m)
    assert set(nearest_team(tracking)) == {v0_says}       # control: in v0 any distance votes
    assert pipeline.NEAR_M == 3.0
    assert set(pipeline.near_team(tracking, pipeline.NEAR_M)) == {near_says}


def test_near_team_reads_its_distance_and_needs_a_tracked_ball():
    tracking = beside_the_ball(4, 3.1, None)
    tracking.ball[2] = np.nan
    assert pipeline.near_team(tracking, 3.0).tolist() == [NOBODY] * 4
    assert pipeline.near_team(tracking, 3.2).tolist() == [HOME, HOME, NOBODY, HOME]


def near_then_far(n_frames=300, until=100):
    """The ball on the centre spot. A home dot 1 m from it before frame `until`, untracked after;
    an away dot 10 m from it from `until` on, untracked before."""
    tracking = beside_the_ball(n_frames, 1.0, 10.0)
    tracking.players[until:, 0] = np.nan
    tracking.players[:until, 1] = np.nan
    return tracking


def raw_vote(tracking):
    """The vote over the near dots alone, before the frames where it has nobody are filled."""
    return smooth(pipeline.near_team(tracking, pipeline.NEAR_M, pipeline.FAST_M_S),
                  pipeline.WINDOW)


def test_the_base_is_the_same_vote_over_the_near_dots_only():
    tracking = near_then_far()
    assert text(pipeline.label_v0(tracking)) == "H100 A200"   # control: in v0 the far dot votes
    # the home dot's last vote (frame 99) is inside the window of every frame up to 149; after
    # that nobody near votes in the whole window, and the home team keeps the ball: no new D
    assert text(raw_vote(tracking)) == "H150 D150"            # control: the vote alone says D
    assert text(pipeline.label_base(tracking)) == "H300"


def near_far_near(n_far, n_near=200, far_m=10.0):
    """The ball on the centre spot. A home dot 1 m from it for n_near frames, then only an away
    dot far_m metres from it for n_far frames, then the home dot 1 m from it again for n_near."""
    tracking = beside_the_ball(2 * n_near + n_far, 1.0, far_m)
    tracking.players[n_near:n_near + n_far, 0] = np.nan
    tracking.players[:n_near, 1] = np.nan
    tracking.players[n_near + n_far:, 1] = np.nan
    return tracking


@pytest.mark.parametrize("n_far, v0_says", [
    (101, "H200 A101 H200"),          # the vote has nobody on one frame only (frame 250)
    (102, "H200 A102 H200"),
    (150, "H200 A150 H200"),
    (226, "H200 A226 H200"),          # 126 frames without a vote: longer than P1's 5 s
    (400, "H200 A400 H200"),
])
def test_the_ball_stays_with_the_team_that_had_it_where_nobody_votes(n_far, v0_says):
    tracking = near_far_near(n_far)
    n = len(tracking.ball)
    v0 = pipeline.label_v0(tracking)
    assert text(v0) == v0_says                                # control: in v0 the far dot votes
    nobody = n_far - 2 * pipeline.WINDOW
    assert text(raw_vote(tracking)) == "H250 D{} H250".format(nobody)   # control: a stretch
    base = pipeline.label_base(tracking)                      # with no vote between two of H
    assert text(base) == "H{}".format(n)                      # no spell of the far dot's team
    assert text(pipeline.label(tracking)) == "H{}".format(n)  # nor in the program
    np.testing.assert_array_equal(base == "D", v0 == "D")     # no D added, none lost


def test_a_stretch_with_no_letter_before_it_takes_the_one_after():
    # the file starts with only a far away dot; the home dot comes near on frame 200
    tracking = beside_the_ball(400, 1.0, 10.0)
    tracking.players[:200, 0] = np.nan
    tracking.players[200:, 1] = np.nan
    assert text(pipeline.label_v0(tracking)) == "A200 H200"   # control
    assert text(raw_vote(tracking)) == "D150 H250"            # control: nobody before frame 150
    assert text(pipeline.label_base(tracking)) == "H400"
    # after a dead ball: the ball lost on 0-149 (v0 says D on 0-99), then only a far away dot
    # until the home dot comes near on frame 350; the base's letter before the stretch is D
    tracking = beside_the_ball(550, 1.0, 10.0)
    tracking.ball[:150] = np.nan
    tracking.players[:350, 0] = np.nan
    tracking.players[350:, 1] = np.nan
    v0 = pipeline.label_v0(tracking)
    assert text(v0) == "D100 A250 H200"                       # control
    assert text(raw_vote(tracking)) == "D300 H250"            # control
    base = pipeline.label_base(tracking)
    assert text(base) == "D100 H450"
    np.testing.assert_array_equal(base == "D", v0 == "D")


def test_a_stretch_with_no_letter_on_either_side_keeps_v0s():
    tracking = beside_the_ball(300, None, 10.0)               # only a far away dot, throughout
    assert text(raw_vote(tracking)) == "D300"                 # control: nobody votes anywhere
    base = pipeline.label_base(tracking)
    assert text(base) == text(pipeline.label_v0(tracking)) == "A300"
    # and between two dead balls: lost, a far away dot, lost again
    tracking = beside_the_ball(500, None, 10.0)
    tracking.ball[:150] = np.nan
    tracking.ball[350:] = np.nan
    v0 = pipeline.label_v0(tracking)
    assert text(v0) == "D100 A300 D100"                       # control
    np.testing.assert_array_equal(pipeline.label_base(tracking), v0)


def test_the_base_has_exactly_v0s_dead_frames(tmp_path):
    raw = str(tmp_path / "atd.txt")
    team = np.array([0, 1, 1], dtype=np.int8)
    random_atd(raw, 2000, len(team), seed=11)
    ball, players = read_atd(raw, team, 2000)
    ball[600:900] = np.nan                                    # a long gap: v0 says D inside it
    tracking = Tracking(ball, players, team, PITCH)
    v0 = pipeline.label_v0(tracking)
    with_d = raw_vote(tracking)
    base = pipeline.label_base(tracking)
    assert (v0 == "D").sum() > 100                            # control: v0 has dead frames
    assert ((with_d == "D") & (v0 != "D")).sum() > 100        # control: the vote alone adds some
    assert not ((with_d != "D") & (v0 == "D")).any()          # ... and keeps every one of v0's
    np.testing.assert_array_equal(base == "D", v0 == "D")
    np.testing.assert_array_equal(base[with_d != "D"], with_d[with_d != "D"])


def test_label_runs_over_the_base_unless_both_rules_are_off():
    tracking = near_then_far()                                # v0 H100 A200; the base differs
    raw_v0 = smooth(nearest_team(tracking), 50)
    params = inspect.signature(pipeline.label).parameters
    assert params["near_m"].default == pipeline.NEAR_M == 3.0           # wired
    assert params["fast_m_s"].default == pipeline.FAST_M_S == 8.0
    assert "near_dead" not in params and "throw_ins" not in params      # taken out
    assert params["arrival"].default is True                            # the last stage, wired
    assert params["restart_near"].default is True                       # P3's voter, wired
    off = {}
    pipeline.label(tracking, near_m=None, fast_m_s=None, stages=off)
    np.testing.assert_array_equal(off["v0"], raw_v0)
    np.testing.assert_array_equal(off[pipeline.BASE], raw_v0)   # both off: the rules receive v0
    np.testing.assert_array_equal(off[pipeline.BEFORE], raw_v0)
    written = dead_ball.dead_mask(tracking)
    for kwargs in ({}, dict(fast_m_s=None)):
        asked = {}
        pipeline.label(tracking, stages=asked, **kwargs)
        np.testing.assert_array_equal(asked["v0"], raw_v0)             # stage v0 stays smooth50
        base = pipeline.label_base(tracking, **kwargs)
        np.testing.assert_array_equal(asked[pipeline.BASE], base)
        assert (base != raw_v0).sum() >= 50                            # control: the base differs
        free = ~(written | asked[pipeline.FLIGHT] | asked[pipeline.KEEPER])
        assert free.all()                                              # control: no rule fires
        np.testing.assert_array_equal(asked[pipeline.BEFORE], base)    # the rules are over it


def rules_over(labels, tracking, tags=None, names=None, arrival=False, restart_near=False):
    """Every rule of the program written over `labels`, stage by stage as label() runs them:
    the dead-ball rules, F1, F2, then P4, P1, P3 (with restart_near, over the base's voter) and
    P5; with arrival, then the change on arrival over the base's voter."""
    labels = labels.copy()
    dead = np.zeros(len(labels), dtype=bool)
    for _, mask in dead_ball.dead_rules(tracking, tags):
        dead |= mask
    labels[dead] = "D"
    labels[dead_ball.in_flight(tracking) & ~dead & (labels == "D")] = "X"
    held = dead_ball.keeper_holding(tracking)
    keeper = (held != NOBODY) & ~dead
    labels[(held == HOME) & ~dead] = "H"
    labels[(held == AWAY) & ~dead] = "A"
    labels, _ = control.gap_edges(labels, tracking, keeper, pipeline.WINDOW)
    labels, _ = control.control(labels, tracking, keeper)
    votes = pipeline.near_team(tracking, pipeline.NEAR_M, pipeline.FAST_M_S)
    restarts = pipeline.tagged_restarts(tracking, tags, names)
    labels, _ = control.restart_team(labels, tracking, restarts, keeper,
                                     votes=votes if restart_near else None)
    labels, _ = control.islands(labels, tracking, keeper)
    if arrival:
        labels, _ = control.change_on_arrival(labels, tracking, votes, keeper, pipeline.WINDOW)
    return labels


def test_the_banked_program_is_the_rules_written_over_v0():
    # the home team near the ball, then only a far away dot for 6 s, the home team again, the
    # ball off the pitch for 2 s, the home team, the away team near for 3.2 s, the home team
    tracking = beside_the_ball(900, 1.0, 10.0)
    tracking.players[:, 1] = np.nan
    tracking.players[200:350, 0] = np.nan
    tracking.players[200:350, 1] = (0.5, 0.5 + 10.0 / PITCH[1])
    tracking.ball[550:600] = (-0.5 / PITCH[0], 0.5)
    tracking.players[700:780, 0] = np.nan
    tracking.players[700:780, 1] = (0.5, 0.5 + 1.0 / PITCH[1])
    v0, base = pipeline.label_v0(tracking), pipeline.label_base(tracking)
    assert text(v0) == "H200 A150 H350 A80 H120"              # control: the far dot votes in v0
    assert text(base) == "H700 A80 H120"                      # control: the base differs
    stages = {}
    banked = pipeline.label(tracking, near_m=None, fast_m_s=None, stages=stages)
    assert stages["ball off the pitch"].sum() == 50           # control: rule C fires
    assert stages[pipeline.CONTROL].sum() == 80               # control: so does P1
    np.testing.assert_array_equal(banked, rules_over(v0, tracking))
    assert text(banked) == "H200 A150 H200 D50 H300"
    program = pipeline.label(tracking)
    np.testing.assert_array_equal(program, rules_over(base, tracking, arrival=True,
                                                      restart_near=True))
    np.testing.assert_array_equal(program, rules_over(base, tracking, arrival=True))
    np.testing.assert_array_equal(program, rules_over(base, tracking))   # no change of team
    assert text(program) == "H550 D50 H300"                   # the program with its defaults


def a_pass_to_away(n_frames=500):
    """The ball on the centre spot. A home dot 1 m from it before frame 200, 10 m from it after;
    an away dot untracked before frame 240, 5 m from it up to frame 280 (on its way, too far to
    have the ball), 1 m from it from frame 280 on (it has the ball)."""
    tracking = beside_the_ball(n_frames, 1.0, 1.0)
    tracking.players[200:, 0] = (0.5 + 10.0 / PITCH[0], 0.5)
    tracking.players[:240, 1] = np.nan
    tracking.players[240:280, 1] = (0.5, 0.5 + 5.0 / PITCH[1])
    return tracking


def test_a_change_of_team_waits_for_the_new_team_to_have_the_ball_near():
    tracking = a_pass_to_away()
    off = {}
    assert text(pipeline.label(tracking, arrival=False, stages=off)) == "H240 A260"   # control
    assert not off[pipeline.ARRIVAL].any()
    stages = {}
    assert text(pipeline.label(tracking, stages=stages)) == "H280 A220"   # the program
    assert stages[pipeline.ARRIVAL].sum() == 40
    assert set(np.flatnonzero(stages[pipeline.ARRIVAL])) == set(range(240, 280))


def test_the_change_on_arrival_is_off_on_the_earlier_programs_path():
    tracking = a_pass_to_away()
    stages = {}
    old = pipeline.label(tracking, near_m=None, fast_m_s=None, stages=stages)
    assert not stages[pipeline.ARRIVAL].any()
    np.testing.assert_array_equal(old, pipeline.label(tracking, near_m=None, fast_m_s=None,
                                                      arrival=False))
    v0 = pipeline.label_v0(tracking)
    np.testing.assert_array_equal(old, rules_over(v0, tracking))   # still the rules over v0
    assert text(old) == "H240 A260"
    # control: the stage would move this change if it ran on that path
    votes = pipeline.near_team(tracking, pipeline.NEAR_M, pipeline.FAST_M_S)
    moved, _ = control.change_on_arrival(old, tracking, votes, np.zeros(len(old), dtype=bool),
                                         pipeline.WINDOW)
    assert text(moved) == "H280 A220"


def test_label_passes_the_speed_test_to_the_base():
    tracking = a_pass_past(0.4)                               # a 10 m/s pass past an away dot
    assert text(pipeline.label(tracking)) == "H480"           # the speed test on: no away spell
    assert text(pipeline.label(tracking, fast_m_s=None)) == "H200 A80 H200"   # off: he wins it
    assert text(pipeline.label(tracking, fast_m_s=12.0)) == "H200 A80 H200"   # above the ball


@pytest.mark.parametrize("home_m, votes", [
    (2.97, HOME),             # 3 cm inside 3 m: he votes
    (3.03, NOBODY),           # 3 cm outside: he does not
])
def test_the_distance_test_at_its_edge(home_m, votes):
    tracking = beside_the_ball(5, home_m, None)
    assert set(nearest_team(tracking)) == {HOME}              # control: in v0 he votes
    assert set(pipeline.near_team(tracking, pipeline.NEAR_M)) == {votes}


@pytest.mark.parametrize("metres_a_second, votes", [
    (7.8, AWAY),              # 0.2 m/s under 8 m/s: the dot beside the ball votes
    (8.2, NOBODY),            # 0.2 m/s over: the ball is in flight, nobody votes
])
def test_the_speed_test_at_its_edge(metres_a_second, votes):
    tracking = riding(40, metres_a_second / 25.0)
    speed = pipeline.ball_speed(tracking)
    np.testing.assert_allclose(speed[5:35], metres_a_second)  # control: the speed it reads
    assert set(nearest_team(tracking)) == {AWAY}              # control: in v0 he votes
    assert set(pipeline.near_team(tracking, None, pipeline.FAST_M_S)[5:35]) == {votes}


@pytest.mark.parametrize("blocks, metres_a_frame, unknown", [
    ([], 0.4, list(range(5)) + list(range(95, 100))),                 # 10 m/s; the two ends
    ([(1, 0, 100)], 0.4, list(range(5)) + list(range(95, 100))),
    ([(1, 0, 50), (2, 50, 100)], 0.4,                                 # never across a period
     list(range(5)) + list(range(45, 55)) + list(range(95, 100))),
    ([(1, 0, 100)], 0.2, list(range(5)) + list(range(95, 100))),      # 5 m/s
    ([(1, 10, 90)], 0.4, list(range(15)) + list(range(85, 100))),     # outside the periods
])
def test_ball_speed_is_its_displacement_over_a_fifth_of_a_second_each_side(
        blocks, metres_a_frame, unknown):
    ball = np.column_stack((0.1 + np.arange(100) * metres_a_frame / PITCH[0], np.full(100, 0.5)))
    tracking = Tracking(ball, np.full((100, 1, 2), np.nan), np.array([0], dtype=np.int8), PITCH,
                        blocks)
    assert dead_ball.SPEED_HALF_S == 0.2                      # +/- 5 frames at 25 fps
    speed = pipeline.ball_speed(tracking)
    known = np.setdiff1d(np.arange(100), unknown)
    assert np.isnan(speed[unknown]).all()
    np.testing.assert_allclose(speed[known], metres_a_frame * 25)


def test_ball_speed_needs_the_ball_at_both_ends():
    ball = np.column_stack((0.1 + np.arange(100) * 0.4 / PITCH[0], np.full(100, 0.5)))
    ball[50] = np.nan
    tracking = Tracking(ball, np.full((100, 1, 2), np.nan), np.array([0], dtype=np.int8), PITCH)
    speed = pipeline.ball_speed(tracking)
    assert np.isnan(speed[[45, 55]]).all()                    # an end on the untracked frame
    np.testing.assert_allclose(speed[[44, 46, 50, 54, 56]], 10.0)   # the frame itself not needed


def riding(n_frames, metres_a_frame):
    """The ball moving along x at metres_a_frame (25 a second) from x = 0.1, and one away dot
    riding 1 m beside it."""
    ball = np.column_stack((0.1 + np.arange(n_frames) * metres_a_frame / PITCH[0],
                            np.full(n_frames, 0.5)))
    players = (ball + (0.0, 1.0 / PITCH[1]))[:, None, :]
    return Tracking(ball, players, np.array([1], dtype=np.int8), PITCH)


@pytest.mark.parametrize("metres_a_frame, fast_m_s, votes_in_the_middle", [
    (0.24, None, AWAY),       # 6 m/s, no speed test: he votes
    (0.24, 8.0, AWAY),        # 6 m/s is not in flight: he votes
    (0.4, 8.0, NOBODY),       # 10 m/s is: nobody votes
    (0.4, None, AWAY),        # ... unless the speed test is off
    (0.4, 12.0, AWAY),        # ... or set above the ball's speed
    (0.36, 8.0, NOBODY),      # 9 m/s
])
def test_a_dot_does_not_vote_while_the_ball_is_fast(metres_a_frame, fast_m_s, votes_in_the_middle):
    tracking = riding(40, metres_a_frame)
    assert set(nearest_team(tracking)) == {AWAY}              # control: in v0 he always votes
    for near_m in (pipeline.NEAR_M, None):                    # he is within 3 m: no difference
        votes = pipeline.near_team(tracking, near_m, fast_m_s)
        assert set(votes[5:35]) == {votes_in_the_middle}
        assert set(votes[:5]) == set(votes[35:]) == {AWAY}    # speed unknown: he votes


def a_pass_past(metres_a_frame, n_fast=80, n_slow=200):
    """The home team holds the ball (standing still, a home dot 1 m from it) for n_slow frames,
    plays it along x at metres_a_frame for n_fast frames with an away dot riding 1 m beside it
    (no home dot tracked meanwhile), and holds it again where it stops for n_slow frames."""
    n = 2 * n_slow + n_fast
    x = 0.3 + np.clip(np.arange(n) - n_slow + 1, 0, n_fast) * metres_a_frame / PITCH[0]
    ball = np.column_stack((x, np.full(n, 0.5)))
    players = np.full((n, 2, 2), np.nan)
    flight = np.zeros(n, dtype=bool)
    flight[n_slow:n_slow + n_fast] = True
    players[~flight, 0] = ball[~flight] + (0.0, 1.0 / PITCH[1])
    players[flight, 1] = ball[flight] + (0.0, 1.0 / PITCH[1])
    return Tracking(ball, players, np.array([0, 1], dtype=np.int8), PITCH)


@pytest.mark.parametrize("metres_a_frame, fast_m_s, the_away_spell_stays", [
    (0.4, 8.0, False),        # 10 m/s: the pass flies past him, the home team keeps it
    (0.4, None, True),        # control: without the speed test his votes win the window
    (0.4, 12.0, True),
    (0.16, 8.0, True),        # 4 m/s: a ball he runs with is his team's
])
def test_a_fast_pass_past_an_opponent_is_not_a_change_of_team(
        metres_a_frame, fast_m_s, the_away_spell_stays):
    tracking = a_pass_past(metres_a_frame)
    v0 = pipeline.label_v0(tracking)
    assert "A" in set(v0[200:280])                            # control: in v0 the away dot wins
    base = pipeline.label_base(tracking, fast_m_s=fast_m_s)
    assert ("A" in set(base)) is the_away_spell_stays
    if not the_away_spell_stays:
        assert set(base) == {"H"}
    np.testing.assert_array_equal(base == "D", v0 == "D")     # no D of its own


def test_a_long_flight_without_a_vote_adds_no_d():
    # a 6 s flight (60 m, on the pitch): from frame 152 to 195 the window holds no vote at all.
    # The away dot votes on the pass's first two frames, while the ball is still slow, so the
    # base's own letter straight before that stretch is A, and the stretch takes it.
    tracking = a_pass_past(0.4, n_fast=150, n_slow=100)
    v0 = pipeline.label_v0(tracking)
    with_d = raw_vote(tracking)
    without = pipeline.label_base(tracking)
    assert text(with_d) == "H149 A3 D44 A7 H147"              # control: no vote in the window
    assert text(without) == "H149 A54 H147"                   # the letter before the stretch
    assert "D" not in set(without) and "D" not in set(v0)
    np.testing.assert_array_equal(without[with_d != "D"], with_d[with_d != "D"])
    labels = pipeline.label(tracking)
    assert "D" not in set(labels)                             # and the program adds none


def ball_out_tracking(n_frames, out_frames, blocks=None, kickoffs=None):
    """One home player beside the ball; the ball 50 cm past the goal line on out_frames."""
    ball = np.tile([0.5, 0.5], (n_frames, 1))
    ball[out_frames] = (-0.5 / PITCH[0], 0.5)
    players = np.tile([0.5, 0.5], (n_frames, 1, 1))
    return Tracking(ball, players, np.array([0], dtype=np.int8), PITCH, blocks or [], kickoffs)


def test_dead_ball_overlay_comes_after_smoothing():
    tracking = ball_out_tracking(300, slice(100, 140))
    assert set(pipeline.label_v0(tracking)[100:140]) == {"H"}   # control: v0 smooths it to H
    labels = pipeline.label(tracking)
    assert "".join(labels[95:145]) == "H" * 5 + "D" * 40 + "H" * 5   # exact edges, no blur


def test_label_writes_d_before_kickoff_and_not_after():
    tracking = ball_out_tracking(100, slice(0, 0), blocks=[(1, 0, 100)], kickoffs={1: 30})
    labels = pipeline.label(tracking)
    assert set(labels[:30]) == {"D"} and set(labels[30:]) == {"H"}


def with_runners(n_frames, start):
    """(players, team): one home player on the centre spot, plus one home and one away player far
    from the ball who stand still and then run along x at 3 m/s from `start` (the players' sign
    of a restart, the restart test's (b))."""
    players = np.tile(np.array([(0.5, 0.5), (0.2, 0.1), (0.8, 0.9)]), (n_frames, 1, 1))
    players[:, 1:, 0] += (np.clip(np.arange(n_frames) - start, 0, None) * 3.0 / 25 /
                          PITCH[0])[:, None]
    return players, np.array([0, 0, 1], dtype=np.int8)


def test_label_writes_the_restart_rule_only_with_tags():
    still, kick = (0.5, 0.5), 0.4 / PITCH[0]               # 10 m/s along x at 25 fps
    ball = np.array([(0.3 + kick * (k + 1), 0.5) for k in range(40)] + [(np.nan, np.nan)] * 50 +
                    [still] * 31 + [(0.5 + kick * (k + 1), 0.5) for k in range(30)])
    tracking = Tracking(ball, *with_runners(len(ball), 120), PITCH)
    throw_in = {"code": "SET PIECES", "start_frame": 60, "end_frame": 140,
                "tags": {"Type": "Throw-in"}}
    np.testing.assert_array_equal(pipeline.label(tracking), pipeline.label_v0(tracking))
    labels = pipeline.label(tracking, [throw_in])
    assert set(labels[40:120]) == {"D"}                     # lost on 40, kicked from 120
    np.testing.assert_array_equal(labels[120:], pipeline.label_v0(tracking)[120:])


def flight_then_throw_in(n_frames=141):
    """Kicked from frame 19 at 10 m/s, lost in flight on 30-79, back inside standing still on 80,
    the throw-in kicked from frame 110; still after that up to n_frames."""
    kick = 0.4 / PITCH[0]
    ball = ([(0.3, 0.5)] * 20 + [(0.3 + kick * (k + 1), 0.5) for k in range(10)] +
            [(np.nan, np.nan)] * 50 + [(0.5, 0.5)] * 31 +
            [(0.5 + kick * (k + 1), 0.5) for k in range(30)])
    ball += [ball[-1]] * (n_frames - len(ball))
    return np.array(ball)


THROW_IN = {"code": "SET PIECES", "start_frame": 60, "end_frame": 200, "tags": {"Type": "Throw-in"}}


def test_a_ball_in_flight_never_writes_over_a_tagged_stoppage():
    # a review fix: with the ball lost, the tag's stoppage outranks F1's guess
    ball = flight_then_throw_in()
    tracking = Tracking(ball, *with_runners(len(ball), 110), PITCH)
    assert dead_ball.in_flight(tracking)[30:80].all()        # control: F1 sees the flight
    labels = pipeline.label(tracking, [THROW_IN])
    assert set(labels[30:110]) == {"D"}          # rule D from the loss (30) to the throw-in
    np.testing.assert_array_equal(labels[110:], pipeline.label_v0(tracking)[110:])


def keeper_gap(extra=None):
    """(ball, players, team): a shot caught by the home keeper on frame 40, the ball lost on
    40-189, his kick from 190 (F2's case), runners who start at the kick; plus `extra` players
    (a list of (team, (x, y))) standing still."""
    keeper, kick = (0.04, 0.5), 0.4 / PITCH[0]
    shot = [(keeper[0] + kick * (20 - k), 0.5) for k in range(1, 21)]
    ball = np.array([(keeper[0] + 21 * kick, 0.5)] * 20 + shot + [(np.nan, np.nan)] * 150 +
                    [(keeper[0] + kick * k, 0.5) for k in range(40)])
    spots = [keeper, (0.3, 0.5), (0.6, 0.4), (0.7, 0.6)] + [xy for _, xy in extra or []]
    players = np.tile(np.array(spots), (len(ball), 1, 1))
    team = [0, 0, 1, 1] + [side for side, _ in extra or []]
    players, team = (np.concatenate((players, with_runners(len(ball), 190)[0][:, 1:]), axis=1),
                     np.array(team + [0, 1], dtype=np.int8))
    return ball, players, team


def test_a_keeper_gap_inside_a_tagged_stoppage_stays_dead():
    # the same fix: a set-piece tag around the gap; its restart is the keeper's kick (190)
    tracking = Tracking(*keeper_gap(), PITCH)
    assert set(pipeline.label(tracking)[40:190]) == {"H"}     # control: F2 alone gives it to H
    free_kick = {"code": "SET PIECES", "start_frame": 60, "end_frame": 230,
                 "tags": {"Type": "Free Kick"}}
    labels = pipeline.label(tracking, [free_kick])
    assert set(labels[40:190]) == {"D"}


def test_a_keeper_gap_with_a_huddle_stays_dead_where_the_huddle_is():
    # the same fix: three of each team bunched and still mid-pitch while the ball is lost
    huddle = [(side, (0.5 + 0.01 * (k % 3), 0.5 + 0.02 * (k // 3))) for k, side in
              enumerate([0, 0, 0, 1, 1, 1])]
    tracking = Tracking(*keeper_gap(huddle), PITCH)
    assert dead_ball.huddle(tracking)[40:190].all()          # control: F3 sees the huddle
    assert dead_ball.keeper_holding(tracking)[40:190].tolist() == [0] * 150   # ... F2 the catch
    assert set(pipeline.label(tracking)[40:190]) == {"D"}


def test_the_lost_ball_corrections_never_undo_before_kickoff():
    ball = flight_then_throw_in(200)
    tracking = Tracking(ball, *with_runners(len(ball), 110), PITCH, [(1, 0, 200)], {1: 150})
    labels = pipeline.label(tracking, [THROW_IN])
    assert set(labels[:150]) == {"D"}            # warm-up kicks before kickoff stay dead (rule B)


def test_label_gives_a_ball_the_keeper_holds_to_his_team():
    keeper, kick = (0.04, 0.5), 0.4 / PITCH[0]
    shot = [(keeper[0] + kick * (20 - k), 0.5) for k in range(1, 21)]
    ball = np.array([(keeper[0] + 21 * kick, 0.5)] * 20 + shot + [(np.nan, np.nan)] * 150 +
                    [(keeper[0] + kick * k, 0.5) for k in range(30)])
    players = np.tile(np.array([keeper, (0.3, 0.5), (0.6, 0.4), (0.7, 0.6)]), (len(ball), 1, 1))
    tracking = Tracking(ball, players, np.array([0, 0, 1, 1], dtype=np.int8), PITCH)
    assert "D" in set(pipeline.label_v0(tracking)[40:190])     # control: v0 calls the gap dead
    assert set(pipeline.label(tracking)[40:190]) == {"H"}      # held by the home keeper


def test_label_leaves_its_tracking_unchanged(tmp_path):
    # no stage moves, drops or recolours a dot, or touches the ball
    raw = str(tmp_path / "atd.txt")
    team = np.array([0, 1, 1], dtype=np.int8)
    random_atd(raw, 600, len(team), seed=5)
    for tracking, tags in ((Tracking(*read_atd(raw, team, 600), team, PITCH, [(1, 0, 600)]), None),
                           (Tracking(*keeper_gap(), PITCH), None),
                           (Tracking(flight_then_throw_in(), *with_runners(141, 110), PITCH),
                            [THROW_IN])):
        before = (tracking.ball.copy(), tracking.players.copy(), tracking.team.copy(),
                  list(tracking.blocks))
        pipeline.label(tracking, tags)                                    # the program
        pipeline.label(tracking, tags, near_m=None, fast_m_s=None)        # over v0 itself
        pipeline.label(tracking, tags, fast_m_s=None)
        pipeline.label(tracking, tags, arrival=False)                     # the last stage off
        pipeline.label(tracking, tags, restart_near=False)                # P3 without the voter
        pipeline.label(tracking, tags, names=(HOME_CLUB, AWAY_CLUB))      # P3 on, with names
        for kept, now in zip(before, (tracking.ball, tracking.players, tracking.team,
                                      tracking.blocks)):
            np.testing.assert_array_equal(kept, now)


def test_label_keeps_v0_first_and_hands_f2s_frames_to_the_possession_stage(monkeypatch):
    tracking = Tracking(*keeper_gap(), PITCH)
    seen = {}
    real = pipeline.control

    def spy(labels, tracking, protected):
        seen["protected"] = protected.copy()
        return real(labels, tracking, protected)

    monkeypatch.setattr(pipeline, "control", spy)
    stages = {}
    pipeline.label(tracking, stages=stages)
    np.testing.assert_array_equal(stages["v0"], pipeline.label_v0(tracking))
    assert stages[pipeline.KEEPER][40:190].all()              # control: F2 wrote the catch
    np.testing.assert_array_equal(seen["protected"], stages[pipeline.KEEPER])
    assert stages[pipeline.CONTROL].dtype == bool


def test_label_runs_f2_then_gap_edges_control_restart_team_and_islands(monkeypatch):
    tracking = Tracking(*keeper_gap(), PITCH)
    calls, seen = [], {}
    real_gap, real_control, real_restart, real_islands = (
        pipeline.gap_edges, pipeline.control, pipeline.restart_team, pipeline.islands)

    def gap_spy(labels, tracking, protected, window):
        out = real_gap(labels, tracking, protected, window)
        calls.append(("gap edges", labels.copy(), protected.copy(), window, out[0].copy()))
        return out

    def control_spy(labels, tracking, protected):
        out = real_control(labels, tracking, protected)
        calls.append(("control", labels.copy(), protected.copy(), None, out[0].copy()))
        return out

    def restart_spy(labels, tracking, restarts, protected, votes=None):
        out = real_restart(labels, tracking, restarts, protected, votes=votes)
        calls.append(("restart team", labels.copy(), protected.copy(), restarts, out[0].copy()))
        seen["restart votes"] = None if votes is None else votes.copy()
        return out

    def islands_spy(labels, tracking, protected):
        out = real_islands(labels, tracking, protected)
        calls.append(("islands", labels.copy(), protected.copy(), None, out[0].copy()))
        return out

    real_arrival = pipeline.change_on_arrival

    def arrival_spy(labels, tracking, votes, protected, window):
        out = real_arrival(labels, tracking, votes, protected, window)
        calls.append(("arrival", labels.copy(), protected.copy(), (votes.copy(), window),
                      out[0].copy()))
        return out

    monkeypatch.setattr(pipeline, "gap_edges", gap_spy)
    monkeypatch.setattr(pipeline, "control", control_spy)
    monkeypatch.setattr(pipeline, "restart_team", restart_spy)
    monkeypatch.setattr(pipeline, "islands", islands_spy)
    monkeypatch.setattr(pipeline, "change_on_arrival", arrival_spy)
    stages = {}
    labels = pipeline.label(tracking, stages=stages)
    assert [c[0] for c in calls] == ["gap edges", "control", "restart team", "islands",
                                     "arrival"]
    np.testing.assert_array_equal(calls[4][1], calls[3][4])   # it reads islands' output
    np.testing.assert_array_equal(calls[4][3][0], pipeline.near_team(
        tracking, pipeline.NEAR_M, pipeline.FAST_M_S))        # over the base's own voter
    assert calls[4][3][1] == pipeline.WINDOW
    np.testing.assert_array_equal(labels, calls[4][4])        # and is the last stage
    assert set(stages[pipeline.BEFORE][40:190]) == {"H"}       # control: F2's catch is in it
    np.testing.assert_array_equal(calls[0][1], stages[pipeline.BEFORE])   # after F2
    assert calls[0][3] == pipeline.WINDOW
    np.testing.assert_array_equal(calls[1][1], calls[0][4])   # control reads gap_edges' output
    np.testing.assert_array_equal(calls[2][1], calls[1][4])   # restart_team reads control's
    assert calls[2][3] == stages[pipeline.RESTARTS] == []     # no tags: no restarts
    np.testing.assert_array_equal(seen["restart votes"], calls[4][3][0])   # the base's voter
    pipeline.label(tracking, restart_near=False)
    assert seen["restart votes"] is None                      # the switch off: P3 as first built
    pipeline.label(tracking, near_m=None, fast_m_s=None)
    assert seen["restart votes"] is None                      # the earlier program's path
    np.testing.assert_array_equal(calls[3][1], calls[2][4])   # islands read restart_team's
    for call in calls:
        np.testing.assert_array_equal(call[2], stages[pipeline.KEEPER])   # F2's frames, to each


def test_main_passes_the_smart_tagging_to_label(tmp_path, data_dir, monkeypatch):
    import sys

    seen = {}
    label = pipeline.label

    def spy(tracking, tags=None, window=pipeline.WINDOW, stages=None, names=None):
        seen["tags"], seen["names"] = tags, names
        return label(tracking, tags, window, stages, names)

    monkeypatch.setattr(pipeline, "label", spy)
    monkeypatch.setattr(sys, "argv", ["pipeline.py", "--out", str(tmp_path / "Cataltina.csv"),
                                      "--data-dir", data_dir])
    pipeline.main()
    set_pieces = [t for t in seen["tags"] if t["code"] == "SET PIECES"]
    assert [(t["start_frame"], t["end_frame"]) for t in set_pieces] == [(75, 88)]   # 3.0-3.5 s
    assert seen["names"] == (HOME_CLUB, AWAY_CLUB)            # P3 reads which team a tag names


def test_label_gives_a_tagged_restart_to_the_named_team_only_with_tags_and_names():
    ball = flight_then_throw_in()                         # lost 30-79, kicked from 110
    tracking = Tracking(ball, *with_runners(len(ball), 110), PITCH)
    named = dict(THROW_IN, tags={"Type": "Throw-in", "Team": "Hillside Athletic"})
    names = (HOME_CLUB, AWAY_CLUB)
    alone = pipeline.label(tracking, [named])
    assert set(alone[110:]) == {"H"}                      # control: the taker's home dot is nearest
    stages = {}
    labels = pipeline.label(tracking, [named], stages=stages, names=names)
    assert stages[pipeline.RESTARTS] == [(110, "A")]
    assert set(labels[110:]) == {"A"}                     # the named team's from the kick
    np.testing.assert_array_equal(labels[:110], alone[:110])
    assert not stages[pipeline.RESTART][:110].any()
    np.testing.assert_array_equal(pipeline.label(tracking, None, names=names),
                                  pipeline.label(tracking))         # no tags: P3 does nothing
    np.testing.assert_array_equal(pipeline.label(tracking, [THROW_IN], names=names),
                                  pipeline.label(tracking, [THROW_IN]))   # a tag naming nobody


def a_restart_then_a_home_dot_near(n_frames=300, near_from=160):
    """flight_then_throw_in: the ball lost on 30-79, back on the centre spot, kicked from frame
    110 at 10 m/s and still from frame 140, 12 m from the spot. The home dot on the spot takes
    it; runners start at the kick; a fourth dot, home, appears 0.5 m from the still ball on
    `near_from` (the home team has the ball near, without a touch)."""
    ball = flight_then_throw_in(n_frames)
    players, team = with_runners(n_frames, 110)
    extra = np.full((n_frames, 1, 2), np.nan)
    extra[near_from:, 0] = (ball[-1][0], ball[-1][1] + 0.5 / PITCH[1])
    return Tracking(ball, np.concatenate((players, extra), axis=1), np.append(team, 0)
                    .astype(np.int8), PITCH)


AWAY_THROW_IN = dict(THROW_IN, tags={"Type": "Throw-in", "Team": "Hillside Athletic"})


def test_label_ends_a_tagged_restart_where_the_other_team_first_has_the_ball_near():
    tracking = a_restart_then_a_home_dot_near()
    names = (HOME_CLUB, AWAY_CLUB)
    stages = {}
    labels = pipeline.label(tracking, [AWAY_THROW_IN], stages=stages, names=names)
    assert stages[pipeline.RESTARTS] == [(110, "A")]          # control: the away restart, kicked
    assert set(labels[30:110]) == {"D"}                       # control: rule D's stoppage
    assert pipeline.near_team(tracking, pipeline.NEAR_M, pipeline.FAST_M_S)[160] == HOME
    assert text(labels[110:]) == "A50 H140"                   # the home dot near ends it
    off = pipeline.label(tracking, [AWAY_THROW_IN], names=names, restart_near=False)
    assert text(off[110:]) == "A190"                          # without the voter: no touch, no end
    np.testing.assert_array_equal(labels[:110], off[:110])


def test_the_earlier_programs_stay_rebuildable_with_tags():
    tracking = a_restart_then_a_home_dot_near()
    tags, names = [AWAY_THROW_IN], (HOME_CLUB, AWAY_CLUB)
    v0, base = pipeline.label_v0(tracking), pipeline.label_base(tracking)

    def program(**kwargs):
        return pipeline.label(tracking, tags, names=names, **kwargs)

    np.testing.assert_array_equal(program(), rules_over(base, tracking, tags, names,
                                                        arrival=True, restart_near=True))
    # the switch off: the program with the change on arrival, P3 as first built
    np.testing.assert_array_equal(program(restart_near=False),
                                  rules_over(base, tracking, tags, names, arrival=True))
    # ... and the change on arrival off too: the program before it
    np.testing.assert_array_equal(program(restart_near=False, arrival=False),
                                  rules_over(base, tracking, tags, names))
    # v0 itself as the base: neither the voter nor the last stage, whatever the switches say
    np.testing.assert_array_equal(program(near_m=None, fast_m_s=None),
                                  rules_over(v0, tracking, tags, names))
    assert text(program()[110:]) != text(program(restart_near=False)[110:])   # control


def test_pipeline_writes_a_valid_file_and_two_builds_match(tmp_path, data_dir):
    result = check_seals.rebuild_seal(data_dir=data_dir, reference=str(tmp_path / "none.csv"))
    assert result[1], result[2]


def test_pipeline_output_is_valid(tmp_path, data_dir):
    import subprocess
    import sys

    out = str(tmp_path / "Cataltina.csv")
    done = subprocess.run([sys.executable, pipeline.__file__, "--out", out, "--data-dir",
                           data_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert done.returncode == 0
    assert pm.validate_submission(pm.read_label_csv(out), N_FRAMES) is None
    codes = read_codes(out)
    assert set(np.unique(codes)) <= {pm.H, pm.A, pm.D}
    # The synthetic tracking (frames 0-99) lies before the first half: v0 gives the home player
    # there (control), rule B2 writes D over every frame outside the halves.
    tracking = load_competition(data_dir=data_dir)
    assert (pipeline.label_v0(tracking)[:100] == "H").sum() > 0
    assert set(codes[:100]) == {pm.D}
    assert os.path.getsize(out) > 0 and len(sha256_of(out)) == 64
