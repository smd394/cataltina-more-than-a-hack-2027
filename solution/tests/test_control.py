"""P1, the possession rule (control.py), on small invented label strings and tracking.

Most fixtures have no tracked ball at all, so no touch can be seen and only the spell structure
decides; the touch tests place one ball and two players by hand.
"""
import numpy as np
import pytest

import control
from common import FPS
from tracking import AWAY, HOME, NOBODY, Tracking

PITCH = (105.0, 68.0)
SHORT = int(round(control.SHORT_S * FPS))


def spells(*parts):
    """'H' * 300 + 'A' * 100 ... from (letter, frames) pairs, as a label array."""
    return np.array(list("".join(letter * n for letter, n in parts)))


def blind(n, blocks=None):
    """A tracking with no ball and no player tracked: no touch anywhere."""
    return Tracking(np.full((n, 2), np.nan), np.full((n, 2, 2), np.nan),
                    np.array([0, 1], dtype=np.int8), PITCH, blocks or [])


def run(labels, tracking=None, protected=None):
    tracking = blind(len(labels)) if tracking is None else tracking
    protected = np.zeros(len(labels), dtype=bool) if protected is None else protected
    return control.control(labels, tracking, protected)


def text(labels):
    """'H300 A100' run-length text of a label array, for readable assertions."""
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append("{}{}".format(labels[start], i - start))
            start = i
    return " ".join(out)


@pytest.mark.parametrize("middle, expected", [
    (100, "H700"),                                   # short: merged into the team around it
    (SHORT - 1, "H{}".format(600 + SHORT - 1)),      # one frame under SHORT_S: merged
    (SHORT, "H300 A{} H300".format(SHORT)),          # SHORT_S itself is not short: kept
    (SHORT + 1, "H300 A{} H300".format(SHORT + 1)),  # a frame longer: kept
])
def test_a_spell_between_two_of_the_other_team_is_merged_only_when_short(middle, expected):
    labels = spells(("H", 300), ("A", middle), ("H", 300))
    out, changed = run(labels)
    assert text(out) == expected
    assert int(changed.sum()) == (middle if middle < SHORT else 0)


@pytest.mark.parametrize("parts, expected", [
    # the other team before it and a D after it (a touch that puts the ball out): merged (W3b)
    ((("H", 300), ("A", 100), ("D", 100), ("H", 300)), "H400 D100 H300"),
    # a D before it and the other team after it: kept (W3c, measured and not taken)
    ((("H", 300), ("D", 100), ("A", 100), ("H", 300)), "H300 D100 A100 H300"),
    # a D on both sides: nothing to merge it into, never across a D
    ((("H", 300), ("D", 50), ("A", 100), ("D", 50), ("H", 300)), "H300 D50 A100 D50 H300"),
    # the first spell of the period, the other team after it: nothing before it, kept
    ((("A", 100), ("H", 300)), "A100 H300"),
])
def test_a_short_spell_beside_a_dead_ball_follows_step_2(parts, expected):
    out, _ = run(spells(*parts))
    assert text(out) == expected


def test_a_merge_never_crosses_a_period_boundary():
    labels = spells(("H", 300), ("A", 100), ("H", 100), ("A", 300))
    out, _ = run(labels)
    assert text(out) != text(labels)                          # control: one period merges
    out, changed = run(labels, blind(len(labels), [(1, 0, 400), (2, 400, 800)]))
    assert text(out) == text(labels) and not changed.any()


def test_x_and_d_frames_are_never_rewritten():
    labels = spells(("H", 300), ("A", 40), ("X", 20), ("A", 40), ("D", 30), ("H", 300))
    out, changed = run(labels)
    assert text(out) == "H340 X20 H40 D30 H300"               # the short A spell merged (W3b)
    assert (out[labels == "X"] == "X").all() and (out[labels == "D"] == "D").all()
    assert not changed[(labels == "X") | (labels == "D")].any()


def test_a_spell_holding_a_protected_frame_is_kept():
    labels = spells(("H", 300), ("A", 100), ("H", 300))
    protected = np.zeros(len(labels), dtype=bool)
    protected[350] = True                                     # one frame F2 wrote
    out, changed = run(labels, protected=protected)
    assert text(out) == "H300 A100 H300" and not changed.any()


@pytest.mark.parametrize("parts, expected", [
    # two short middles: the shorter is merged first, wherever it is
    ((("H", 300), ("A", 40), ("H", 60), ("A", 300)), "H400 A300"),
    ((("H", 300), ("A", 60), ("H", 40), ("A", 300)), "H300 A400"),
    # the same length: the earlier one
    ((("H", 300), ("A", 50), ("H", 50), ("A", 300)), "H400 A300"),
])
def test_the_shortest_spell_goes_first_and_the_earliest_on_a_tie(parts, expected):
    labels = spells(*parts)
    first, _ = run(labels)
    second, _ = run(labels)
    assert text(first) == expected
    np.testing.assert_array_equal(first, second)              # two calls, identical arrays


def touch_tracking(n, turn, reach_away_m):
    """The ball rolls along x at TOUCH_M_S up to frame `turn`, then along y (a change of direction
    of TOUCH_M_S x sqrt 2); tracked only on turn - 30 .. turn + 30. A home dot stands 0.11 m from
    the ball at the turn (the nearest) and an away dot reach_away_m from it."""
    speed = control.TOUCH_M_S / FPS                          # metres per frame
    ball = np.full((n, 2), np.nan)
    turn_xy = np.array([50.0, 30.0])
    for f in range(turn - 30, turn + 31):
        step = f - turn
        xy = turn_xy + (np.array([step * speed, 0.0]) if step <= 0 else np.array([0.0, step * speed]))
        ball[f] = xy / np.array(PITCH)
    home = (turn_xy + np.array([0.1, -0.05])) / np.array(PITCH)
    away = (turn_xy + np.array([0.0, -reach_away_m])) / np.array(PITCH)
    players = np.tile(np.array([home, away]), (n, 1, 1))
    return Tracking(ball, players, np.array([0, 1], dtype=np.int8), PITCH, [])


def test_a_short_spell_its_team_touched_is_kept_even_with_an_opponent_nearer():
    labels = spells(("H", 300), ("A", 100), ("H", 300))
    tracking = touch_tracking(len(labels), 350, 0.75 * control.REACH_M)
    assert control.touches(tracking)[1][345:356].any()        # control: the away touch is seen
    out, changed = run(labels, tracking)
    assert text(out) == "H300 A100 H300" and not changed.any()


def test_the_same_spell_with_its_player_out_of_reach_at_the_change_is_merged():
    labels = spells(("H", 300), ("A", 100), ("H", 300))
    tracking = touch_tracking(len(labels), 350, control.REACH_M + 0.6)
    assert control.touches(tracking)[0][345:356].any()        # control: the change is seen
    out, changed = run(labels, tracking)
    assert text(out) == "H700" and int(changed.sum()) == 100


def test_the_numbers_default_to_the_params_when_called(monkeypatch):
    labels = spells(("H", 300), ("A", 100), ("H", 300))
    assert text(run(labels)[0]) == "H700"                     # control: merged at SHORT_S
    monkeypatch.setattr(control, "SHORT_S", 100.0 / FPS)      # read at call time, not at import
    assert text(run(labels)[0]) == "H300 A100 H300"
    out, _ = control.control(labels, blind(len(labels)), np.zeros(len(labels), dtype=bool),
                             short_s=101.0 / FPS)
    assert text(out) == "H700"                                # an argument beats the Params


# -- P4, the edges of a long ball gap --------------------------------------------------------------

WINDOW = 50


def gap_tracking(n, gap, blocks=None):
    """The ball tracked (standing still mid-pitch) on every frame but gap = (a, b); no players."""
    ball = np.tile([0.5, 0.5], (n, 1))
    ball[gap[0]:gap[1]] = np.nan
    return Tracking(ball, np.full((n, 2, 2), np.nan), np.array([0, 1], dtype=np.int8), PITCH,
                    blocks or [])


def edges(labels, tracking, protected=None, **kwargs):
    protected = np.zeros(len(labels), dtype=bool) if protected is None else protected
    return control.gap_edges(labels, tracking, protected, WINDOW, **kwargs)


@pytest.mark.parametrize("parts, expected, written", [
    # v0 carried H 50 frames into the gap and A 50 frames back from its end: both become D
    ((("H", 150), ("D", 200), ("A", 150)), "H100 D300 A100", 100),
    # a letter left between two D runs inside the gap becomes D too, not only the edges
    ((("H", 150), ("D", 50), ("H", 40), ("D", 110), ("A", 150)), "H100 D300 A100", 140),
])
def test_every_carried_letter_of_a_long_gap_becomes_d(parts, expected, written):
    # ball lost on 100-399
    labels = spells(*parts)
    out, changed = edges(labels, gap_tracking(500, (100, 400)))
    assert text(out) == expected
    assert int(changed.sum()) == written
    assert (out[:100] == "H").all() and (out[400:] == "A").all()   # nothing outside the gap


@pytest.mark.parametrize("gap, expected", [
    (2 * WINDOW, "H140 D20 A140"),          # exactly 2 x window: left alone
    (2 * WINDOW + 1, "H100 D101 A100"),     # one frame longer: its edges become D
])
def test_a_gap_must_be_longer_than_two_windows(gap, expected):
    lead = 100
    labels = spells(("H", lead + 40), ("D", gap - 80), ("A", 40 + lead))
    out, _ = edges(labels, gap_tracking(len(labels), (lead, lead + gap)))
    assert text(out) == expected


def test_a_gap_without_any_d_or_x_is_left_alone():
    labels = spells(("H", 500))
    out, changed = edges(labels, gap_tracking(500, (100, 400)))
    assert text(out) == "H500" and not changed.any()


def test_gap_edges_never_touch_f2s_frames():
    labels = spells(("H", 150), ("D", 200), ("A", 150))
    protected = np.zeros(500, dtype=bool)
    protected[100:150] = True                                 # F2 wrote the gap's first frames
    out, changed = edges(labels, gap_tracking(500, (100, 400)), protected)
    assert text(out) == "H150 D250 A100"
    assert not changed[protected].any()


def test_gap_edges_never_touch_an_x():
    labels = spells(("H", 120), ("X", 30), ("D", 200), ("A", 150))
    out, changed = edges(labels, gap_tracking(500, (100, 400)))
    assert text(out) == "H100 D20 X30 D250 A100"              # the H inside the gap became D
    assert (out[120:150] == "X").all() and not changed[120:150].any()


def test_gap_edges_never_touch_f2s_frames_between_two_d_runs():
    labels = spells(("H", 150), ("D", 50), ("H", 40), ("D", 110), ("A", 150))
    protected = np.zeros(500, dtype=bool)
    protected[200:240] = True                                 # F2 wrote the letters in between
    out, changed = edges(labels, gap_tracking(500, (100, 400)), protected)
    assert text(out) == "H100 D100 H40 D160 A100"
    assert not changed[protected].any()


def test_gap_edges_stay_inside_their_period():
    labels = spells(("H", 150), ("D", 200), ("A", 150))
    tracking = gap_tracking(500, (100, 400), [(1, 0, 250), (2, 250, 500)])
    out, _ = edges(labels, tracking)                          # two runs of 150, each with a D
    assert text(out) == "H100 D300 A100"
    tracking = gap_tracking(500, (100, 400), [(1, 0, 180), (2, 180, 500)])
    out, _ = edges(labels, tracking)                          # 100-179 is 80 frames: too short
    assert text(out) == "H150 D250 A100"


def test_gap_edges_before_p1_change_which_spell_p1_merges():
    # H, then an A spell of SHORT frames whose last 50 v0 carried into a long gap, the gap, 50
    # carried A frames and 30 tracked A frames, then H. Without P4 the first A spell is SHORT_S
    # long (kept); with P4 first it is shorter and beside a D (merged, W3b)
    tail = 50
    gap = (300 + SHORT - tail, 300 + SHORT + 200 + tail)
    labels = spells(("H", 300), ("A", SHORT), ("D", 200), ("A", tail + 30), ("H", 300))
    tracking = gap_tracking(len(labels), gap)
    alone, _ = run(labels, tracking)
    assert text(alone) == text(labels)                        # control: P1 alone keeps it
    after_p4, _ = edges(labels, tracking)
    out, _ = run(after_p4, tracking)
    assert text(out) == "H{} D{} A30 H300".format(300 + SHORT - tail, 200 + 2 * tail)


def test_gap_edges_leave_their_inputs_unchanged():
    labels = spells(("H", 150), ("D", 200), ("A", 150))
    tracking = gap_tracking(500, (100, 400))
    protected = np.zeros(500, dtype=bool)
    before = (labels.copy(), tracking.ball.copy(), protected.copy())
    out, _ = edges(labels, tracking, protected)
    assert text(out) != text(labels)                          # control: it did write
    for kept, now in zip(before, (labels, tracking.ball, protected)):
        np.testing.assert_array_equal(kept, now)


# -- P3, who has the ball after a tagged restart ------------------------------------------------

def restart_tracking(n, kick, turn, taker_team, toucher_team, toucher_m=0.1):
    """The ball still at (50, 30) m up to `kick`, then rolling along x at 8 m/s, turning along y
    at `turn` (a change of direction). Slot 0, of taker_team, stands on the ball at the kick;
    slot 1, of toucher_team, stands toucher_m from the ball at the turn; slot 2, a home dot,
    stays 30 m away."""
    speed = 8.0 / FPS
    spot = np.array([50.0, 30.0])
    xy = np.zeros((n, 2))
    for f in range(n):
        if f <= kick:
            xy[f] = spot
        elif f <= turn:
            xy[f] = spot + [(f - kick) * speed, 0.0]
        else:
            xy[f] = spot + [(turn - kick) * speed, (f - turn) * speed]
    at_turn = spot + [(turn - kick) * speed, -toucher_m]
    dots = np.array([spot + [0.0, -0.2], at_turn, spot + [0.0, 30.0]]) / np.array(PITCH)
    team = np.array([taker_team, toucher_team, 0], dtype=np.int8)
    return Tracking(xy / np.array(PITCH), np.tile(dots, (n, 1, 1)), team, PITCH, [])


def restart(labels, tracking, restarts, protected=None):
    protected = np.zeros(len(labels), dtype=bool) if protected is None else protected
    return control.restart_team(labels, tracking, restarts, protected)


def test_a_restart_gives_the_ball_to_its_team_up_to_the_other_teams_first_touch():
    labels = spells(("H", 100), ("A", 300))
    tracking = restart_tracking(400, 100, 200, 0, 1)
    out, changed = restart(labels, tracking, [(100, "H")])
    turn = 100 + int(np.flatnonzero(control.touches(tracking)[1][100:])[0])
    assert 190 <= turn <= 200                                 # control: the away touch is seen
    assert text(out) == "H{} A{}".format(turn, 400 - turn)
    assert int(changed.sum()) == turn - 100


def test_the_takers_own_dot_in_the_other_colour_does_not_end_it():
    # the taker's dot shows away (slot 0, team 1) and is on the ball as it is kicked: its "touch"
    # at the kick is not the other team's; the away dot at the turn is
    labels = spells(("H", 100), ("A", 300))
    tracking = restart_tracking(400, 100, 200, 1, 1)
    assert control.touches(tracking)[1][95:106].any()         # control: a touch by an away dot
    assert control.taker(tracking, 100) == 0
    out, _ = restart(labels, tracking, [(100, "H")])
    assert text(out).startswith("H19") or text(out).startswith("H20")
    assert int(np.sum(out == "H")) > 150


def with_dot(tracking, team, xy_m):
    """The same tracking with one more dot, of `team`, standing still at xy_m (metres)."""
    dot = np.tile(np.array(xy_m) / np.array(PITCH), (tracking.n_frames, 1, 1))
    return Tracking(tracking.ball, np.concatenate((tracking.players, dot), axis=1),
                    np.append(tracking.team, team).astype(np.int8), PITCH, tracking.blocks)


def test_a_second_dot_of_the_other_team_at_the_kick_does_not_end_it_at_once():
    # an away dot 1 m from the ball as the home taker (0.2 m) kicks it: for 0.4 s the touch
    # test's window still holds the kick itself, which is not that dot's touch
    labels = spells(("H", 100), ("A", 300))
    tracking = with_dot(restart_tracking(400, 100, 200, 0, 1), 1, (50.0, 31.0))
    assert control.taker(tracking, 100) == 0                  # control: the home dot takes it
    assert control.touches(tracking)[1][100]                  # control: the kick reads as a touch
    out, _ = restart(labels, tracking, [(100, "H")])
    assert text(out).startswith("H19") or text(out).startswith("H20")


def test_a_touch_just_after_the_guard_still_ends_it():
    # the away dot plays the ball 0.6 s after the kick: its touch is seen from 0.48 s, past the
    # 0.4 s guard
    labels = spells(("H", 100), ("A", 300))
    tracking = restart_tracking(400, 100, 115, 0, 1)
    seen = 100 + int(np.flatnonzero(control.touches(tracking)[1][100:])[0])
    assert 100 + int(round(0.4 * FPS)) <= seen < 115          # control: past the guard
    out, changed = restart(labels, tracking, [(100, "H")])
    assert text(out) == "H{} A{}".format(seen, 400 - seen)
    assert int(changed.sum()) == seen - 100


@pytest.mark.parametrize("parts, protect, expected", [
    # no touch at all: the whole spell, up to the D, never past it
    ((("H", 100), ("A", 50), ("D", 50), ("A", 200)), None, "H150 D50 A200"),
    # the spell ends where the restart's own team's letters start again
    ((("H", 100), ("A", 50), ("H", 20), ("A", 230)), None, "H170 A230"),
    # a frame F2 wrote stops it
    ((("H", 100), ("A", 300)), 130, "H130 A270"),
])
def test_a_restart_never_passes_a_d_the_end_of_its_spell_or_a_protected_frame(parts, protect,
                                                                           expected):
    labels = spells(*parts)
    tracking = blind(len(labels))
    tracking.ball[:] = 0.5                                    # tracked, still: no touch anywhere
    protected = np.zeros(len(labels), dtype=bool)
    if protect is not None:
        protected[protect] = True
    out, _ = restart(labels, tracking, [(100, "H")], protected)
    assert text(out) == expected


@pytest.mark.parametrize("parts", [
    (("H", 100), ("H", 300)),                                 # already its team's letter
    (("H", 100), ("D", 10), ("A", 290)),                      # a D at the kick
    (("H", 100), ("X", 10), ("A", 290)),                      # an X at the kick
])
def test_a_restart_already_its_teams_or_dead_changes_nothing(parts):
    labels = spells(*parts)
    out, changed = restart(labels, blind(len(labels)), [(100, "H")])
    assert text(out) == text(labels) and not changed.any()


def test_restart_team_leaves_its_inputs_unchanged():
    labels = spells(("H", 100), ("A", 300))
    tracking = restart_tracking(400, 100, 200, 0, 1)
    protected = np.zeros(400, dtype=bool)
    before = (labels.copy(), tracking.ball.copy(), tracking.players.copy(), protected.copy())
    out, _ = restart(labels, tracking, [(100, "H")], protected)
    assert text(out) != text(labels)                          # control: it did rewrite
    for kept, now in zip(before, (labels, tracking.ball, tracking.players, protected)):
        np.testing.assert_array_equal(kept, now)


# -- P3 with the base's voter: until the other team touches the ball OR is the one near it ------

GUARD = int(round(control.KICK_GUARD_S * FPS))                # 10 frames: 0.4 s at 25 fps


def nobody(n):
    return np.full(n, NOBODY, dtype=np.int8)


def near_votes(n, *stretches):
    """A voter array of n frames: nobody, except (first, stop, "H" / "A") stretches where the
    base's voter names that team."""
    out = nobody(n)
    for first, stop, letter in stretches:
        out[first:stop] = HOME if letter == "H" else AWAY
    return out


def restart_voted(labels, tracking, restarts, votes, protected=None):
    protected = np.zeros(len(labels), dtype=bool) if protected is None else protected
    return control.restart_team(labels, tracking, restarts, protected, votes=votes)


def first_away_touch(tracking, kick):
    return kick + int(np.flatnonzero(control.touches(tracking)[1][kick:])[0])


@pytest.mark.parametrize("vote_at, ends", [
    (170, "vote"),                    # the away dot is the one near the ball before it touches it
    (250, "touch"),                   # it touches the ball first (about frame 195)
    (None, "touch"),                  # no vote at all: the touch, as built
])
def test_the_spell_ends_at_the_earlier_of_the_vote_and_the_touch(vote_at, ends):
    labels = spells(("H", 100), ("A", 300))
    tracking = restart_tracking(400, 100, 200, 0, 1)
    touch = first_away_touch(tracking, 100)
    assert 190 <= touch <= 200                                # control: the away touch is seen
    votes = nobody(400) if vote_at is None else near_votes(400, (vote_at, 400, "A"))
    out, changed = restart_voted(labels, tracking, [(100, "H")], votes)
    end = vote_at if ends == "vote" else touch
    assert text(out) == "H{} A{}".format(end, 400 - end)
    assert int(changed.sum()) == end - 100


def still_ball(n, dots_m, team):
    """The ball tracked and still at (50, 30) m on every frame; dots standing at dots_m (metres,
    per slot), of `team`. No touch anywhere."""
    ball = np.tile(np.array([50.0, 30.0]) / np.array(PITCH), (n, 1))
    dots = np.tile(np.array(dots_m) / np.array(PITCH), (n, 1, 1))
    return Tracking(ball, dots, np.array(team, dtype=np.int8), PITCH, [])


@pytest.mark.parametrize("vote, expected", [
    ((101, 100 + GUARD), "H400"),                     # only inside the first 0.4 s: not counted
    ((100 + GUARD - 1, 100 + GUARD + 1), "H110 A290"),   # its first frame past the guard counts
    ((100 + GUARD, 400), "H110 A290"),
])
def test_a_vote_is_not_counted_in_the_first_kick_guard_seconds(vote, expected):
    # the home taker 0.2 m from the still ball at the kick; from the next frame an away dot
    # stands 0.1 m from it, the nearest dot, so each vote below is that dot's
    labels = spells(("H", 100), ("A", 300))
    tracking = still_ball(400, [(50.0, 29.8), (50.0, 40.0), (50.0, 0.0)], [0, 1, 0])
    tracking.players[101:, 1] = np.array([50.0, 30.1]) / np.array(PITCH)
    assert control.taker(tracking, 100) == 0                  # control: the home dot takes it
    out, _ = restart_voted(labels, tracking, [(100, "H")], near_votes(400, vote + ("A",)))
    assert text(out) == expected


def test_with_the_voter_the_kicks_own_change_is_still_not_the_other_teams_touch():
    # an away dot 1 m from the ball as the home taker kicks it (the touch test reads the kick as
    # its touch for 0.4 s): the spell ends at the away touch at the turn, not at the kick
    labels = spells(("H", 100), ("A", 300))
    tracking = with_dot(restart_tracking(400, 100, 200, 0, 1), 1, (50.0, 31.0))
    assert control.touches(tracking)[1][100]                  # control: the kick reads as a touch
    touch = first_away_touch(tracking, 100 + GUARD)
    assert 190 <= touch <= 200                                # control: the touch at the turn
    out, _ = restart_voted(labels, tracking, [(100, "H")], nobody(400))
    assert text(out) == "H{} A{}".format(touch, 400 - touch)


def test_the_takers_dot_gives_no_vote():
    # the taker's dot shows the other colour (away) and stands 0.2 m from the ball: the voter
    # names away wherever it is the nearest dot, and that is not the other team getting the ball
    labels = spells(("H", 100), ("A", 300))
    tracking = still_ball(400, [(50.0, 29.8), (50.0, 50.0), (50.0, 0.0)], [1, 1, 0])
    assert control.taker(tracking, 100) == 0                  # control: slot 0 takes it
    out, _ = restart_voted(labels, tracking, [(100, "H")], near_votes(400, (0, 400, "A")))
    assert text(out) == "H400"                                # no end: the whole reach
    # control: from frame 200 another away dot stands nearer the ball than the taker
    tracking.players[200:, 1] = np.array([50.0, 30.1]) / np.array(PITCH)
    out, _ = restart_voted(labels, tracking, [(100, "H")], near_votes(400, (0, 400, "A")))
    assert text(out) == "H200 A200"


@pytest.mark.parametrize("parts, votes, expected", [
    # the kick on the restart team's own letter: it now acts, rewriting only the other team's
    # letters from the kick to the end (the vote at 330)
    ((("H", 150), ("A", 100), ("H", 50), ("A", 100)), [(330, 400, "A")], "H330 A70"),
    # no end: up to the reach, through the restart team's own letters
    ((("H", 100), ("A", 50), ("H", 20), ("A", 230)), [], "H400"),
    # the other team's letter at the kick, as before
    ((("H", 100), ("A", 300)), [(250, 400, "A")], "H250 A150"),
])
def test_a_kick_on_any_teams_letter_acts_and_rewrites_only_the_other_teams_letters(parts, votes,
                                                                                    expected):
    labels = spells(*parts)
    out, changed = restart_voted(labels, blind(len(labels)), [(100, "H")],
                                 near_votes(len(labels), *votes))
    assert text(out) == expected
    assert (labels[changed] == "A").all() and (out[changed] == "H").all()


@pytest.mark.parametrize("parts, protect, blocks, expected", [
    ((("H", 100), ("A", 50), ("D", 50), ("A", 200)), None, None, "H150 D50 A200"),   # a D
    ((("H", 100), ("A", 50), ("D", 1), ("A", 249)), None, None, "H150 D1 A249"),     # one D frame
    ((("H", 100), ("A", 300)), 130, None, "H130 A270"),                              # F2's frame
    ((("H", 100), ("A", 300)), None, [(1, 0, 250), (2, 250, 400)], "H250 A150"),     # the period
    ((("H", 100), ("A", 50), ("D", 10), ("X", 10), ("A", 230)), None, None,
     "H150 D10 X10 A230"),                                    # an X after a D reads as the D
])
def test_with_the_voter_the_reach_stops_at_a_d_a_protected_frame_and_the_periods_end(
        parts, protect, blocks, expected):
    labels = spells(*parts)
    protected = np.zeros(len(labels), dtype=bool)
    if protect is not None:
        protected[protect] = True
    out, _ = restart_voted(labels, blind(len(labels), blocks), [(100, "H")],
                           nobody(len(labels)), protected)
    assert text(out) == expected


def test_with_the_voter_an_x_is_never_changed_and_does_not_stop_the_reach():
    labels = spells(("H", 100), ("A", 50), ("X", 10), ("A", 240))
    out, changed = restart_voted(labels, blind(400), [(100, "H")], nobody(400))
    assert text(out) == "H150 X10 H240"                       # read as the A before it
    assert not changed[labels == "X"].any()


@pytest.mark.parametrize("parts", [
    (("H", 100), ("D", 10), ("A", 290)),                      # a D at the kick
    (("H", 100), ("X", 10), ("A", 290)),                      # an X at the kick
])
def test_with_the_voter_a_kick_on_a_d_or_an_x_changes_nothing(parts):
    labels = spells(*parts)
    out, changed = restart_voted(labels, blind(len(labels)), [(100, "H")],
                                 near_votes(len(labels), (200, 400, "A")))
    assert text(out) == text(labels) and not changed.any()


@pytest.mark.parametrize("parts, with_voter, without", [
    # the kick on its own team's letter: nothing without the voter
    ((("H", 150), ("A", 250)), "H400", "H150 A250"),
    # through its own team's letters: without the voter it stops at the end of the other's spell
    ((("H", 100), ("A", 50), ("H", 20), ("A", 230)), "H400", "H170 A230"),
])
def test_without_the_voter_it_is_the_rule_as_first_built(parts, with_voter, without):
    labels = spells(*parts)
    tracking = blind(len(labels))
    tracking.ball[:] = 0.5                                    # tracked, still: no touch anywhere
    assert text(restart_voted(labels, tracking, [(100, "H")], nobody(len(labels)))[0]) == \
        with_voter
    assert text(restart(labels, tracking, [(100, "H")])[0]) == without
    np.testing.assert_array_equal(
        restart(labels, tracking, [(100, "H")])[0],
        control.restart_team(labels, tracking, [(100, "H")], np.zeros(len(labels), dtype=bool),
                             votes=None)[0])


def test_restart_team_with_the_voter_leaves_its_inputs_unchanged():
    labels = spells(("H", 100), ("A", 300))
    tracking = restart_tracking(400, 100, 200, 0, 1)
    votes = near_votes(400, (170, 400, "A"))
    protected = np.zeros(400, dtype=bool)
    before = (labels.copy(), tracking.ball.copy(), tracking.players.copy(), tracking.team.copy(),
              votes.copy(), protected.copy())
    first, _ = restart_voted(labels, tracking, [(100, "H")], votes, protected)
    second, _ = restart_voted(labels, tracking, [(100, "H")], votes, protected)
    assert text(first) == "H170 A230"                         # control: it did rewrite
    np.testing.assert_array_equal(first, second)
    for kept, now in zip(before, (labels, tracking.ball, tracking.players, tracking.team, votes,
                                  protected)):
        np.testing.assert_array_equal(kept, now)


def test_control_leaves_its_inputs_unchanged():
    labels = spells(("H", 300), ("A", 100), ("H", 300))
    tracking = touch_tracking(len(labels), 350, control.REACH_M + 0.6)
    protected = np.zeros(len(labels), dtype=bool)
    before = (labels.copy(), tracking.ball.copy(), tracking.players.copy(), tracking.team.copy(),
              protected.copy())
    out, _ = run(labels, tracking, protected)
    assert text(out) == "H700"                                # control: it did rewrite
    for kept, now in zip(before, (labels, tracking.ball, tracking.players, tracking.team,
                                  protected)):
        np.testing.assert_array_equal(kept, now)


# -- the numbers, pinned with literal metres and seconds (not the Params) -----------------------

def turning_ball(n, at, before_m_s, after_m_s, away_m):
    """The ball tracked on at - 30 .. at + 30 only, moving with velocity before_m_s (m/s, x and y)
    up to frame `at` and after_m_s from it; an away dot away_m metres beside the ball's spot at
    `at` (across the path before it, -y), a home dot 30 m away."""
    spot = np.array([50.0, 30.0])
    ball = np.full((n, 2), np.nan)
    for f in range(at - 30, at + 31):
        step = (f - at) / float(FPS)
        ball[f] = (spot + step * np.array(before_m_s if f <= at else after_m_s)) / np.array(PITCH)
    dots = np.array([spot + [0.0, 30.0], spot + [0.0, -away_m]]) / np.array(PITCH)
    return Tracking(ball, np.tile(dots, (n, 1, 1)), np.array([0, 1], dtype=np.int8), PITCH, [])


@pytest.mark.parametrize("seconds, merged", [
    (4.96, True),                         # just under 5 s: short, merged
    (5.0, False),                         # 5 s: not short, kept
])
def test_short_is_under_five_seconds(seconds, merged):
    middle = int(round(seconds * FPS))
    labels = spells(("H", 300), ("A", middle), ("H", 300))
    out, _ = run(labels)
    assert (text(out) == "H{}".format(600 + middle)) == merged


@pytest.mark.parametrize("away_m, kept", [
    (1.4, True),                          # an away player 1.4 m from the ball as it turns: a touch
    (1.6, False),                         # 1.6 m: out of reach, no touch, the spell is merged
])
def test_reach_is_one_and_a_half_metres(away_m, kept):
    labels = spells(("H", 300), ("A", 100), ("H", 300))
    tracking = turning_ball(700, 350, (6.0, 0.0), (0.0, 6.0), away_m)   # a change of 8.5 m/s
    out, _ = run(labels, tracking)
    assert (text(out) == text(labels)) == kept


@pytest.mark.parametrize("after_m_s, kept", [
    (4.8, False),                         # from 1 m/s to 4.8 m/s: a change of 3.8 m/s, no touch
    (5.2, True),                          # to 5.2 m/s: 4.2 m/s, a touch
])
def test_a_touch_is_a_change_of_four_metres_a_second(after_m_s, kept):
    labels = spells(("H", 300), ("A", 100), ("H", 300))
    tracking = turning_ball(700, 350, (1.0, 0.0), (after_m_s, 0.0), 0.2)
    peak = np.nanmax(control.ball_change(tracking))
    assert abs(peak - (after_m_s - 1.0)) < 1e-6                # control: the change as intended
    out, _ = run(labels, tracking)
    assert (text(out) == text(labels)) == kept


@pytest.mark.parametrize("metres, slot", [
    (9.0, 0),                             # a dot 9 m from the ball at the kick: the taker
    (9.3, None),                          # 9.3 m: beyond 9.15 m, nobody takes it
])
def test_the_taker_is_within_nine_metres_fifteen(metres, slot):
    spot = np.array([50.0, 30.0])
    ball = np.tile(spot / np.array(PITCH), (10, 1))
    dots = np.tile((spot + [metres, 0.0]) / np.array(PITCH), (10, 1, 1))
    tracking = Tracking(ball, dots, np.array([1], dtype=np.int8), PITCH, [])
    assert control.taker(tracking, 5) == slot



# -- P5, a possession under 1 s with a dead ball on both sides ----------------------------------

ONE_S = int(round(1.0 * FPS))


def dead_islands(labels, blocks=None, protected=None):
    protected = np.zeros(len(labels), dtype=bool) if protected is None else protected
    return control.islands(labels, blind(len(labels), blocks), protected)


@pytest.mark.parametrize("parts, expected", [
    # under 1 s between two dead balls: dead
    ((("H", 100), ("D", 50), ("A", ONE_S - 1), ("D", 50), ("H", 100)),
     "H100 D{} H100".format(100 + ONE_S - 1)),
    # exactly 1 s: kept
    ((("H", 100), ("D", 50), ("A", ONE_S), ("D", 50), ("H", 100)),
     "H100 D50 A{} D50 H100".format(ONE_S)),
    # a team's spell on one side: kept (it is P1's business, not P5's)
    ((("H", 100), ("A", 10), ("D", 50), ("H", 100)), "H100 A10 D50 H100"),
    ((("H", 100), ("D", 50), ("A", 10), ("H", 100)), "H100 D50 A10 H100"),
    # the start and the end of the period count as a dead ball
    ((("A", 10), ("D", 50), ("H", 100)), "D60 H100"),
    ((("H", 100), ("D", 50), ("A", 10)), "H100 D60"),
])
def test_a_possession_under_one_second_between_two_dead_balls_is_dead(parts, expected):
    out, changed = dead_islands(spells(*parts))
    assert text(out) == expected
    assert int(changed.sum()) == int(np.sum(out != spells(*parts)))


def test_an_island_holding_a_frame_f2_wrote_is_kept():
    labels = spells(("H", 100), ("D", 50), ("A", 10), ("D", 50), ("H", 100))
    assert text(dead_islands(labels)[0]) == "H100 D110 H100"  # control: dead without F2
    protected = np.zeros(len(labels), dtype=bool)
    protected[155] = True
    out, changed = dead_islands(labels, protected=protected)
    assert text(out) == text(labels) and not changed.any()


def test_an_islands_x_frames_stay_x():
    # the scorer reads A6 X4 as one spell of 10 frames; its X frames are not rewritten
    labels = spells(("H", 100), ("D", 50), ("A", 6), ("X", 4), ("D", 50), ("H", 100))
    out, changed = dead_islands(labels)
    assert text(out) == "H100 D56 X4 D50 H100"
    assert not changed[labels == "X"].any()


def test_islands_are_read_inside_each_period():
    # an A spell under 1 s right after halftime with a D after it: the period's start is its edge
    labels = spells(("H", 100), ("A", 10), ("D", 50), ("H", 100))
    assert text(dead_islands(labels)[0]) == text(labels)      # control: one period, kept
    out, _ = dead_islands(labels, [(1, 0, 100), (2, 100, 260)])
    assert text(out) == "H100 D60 H100"


def test_islands_leave_their_inputs_unchanged_and_two_calls_agree():
    labels = spells(("H", 100), ("D", 50), ("A", 10), ("D", 50), ("H", 100))
    protected = np.zeros(len(labels), dtype=bool)
    before = (labels.copy(), protected.copy())
    first, _ = dead_islands(labels, protected=protected)
    second, _ = dead_islands(labels, protected=protected)
    assert text(first) != text(labels)                        # control: it did write
    np.testing.assert_array_equal(first, second)
    for kept, now in zip(before, (labels, protected)):
        np.testing.assert_array_equal(kept, now)


# -- the change of team written on the ball's arrival ---------------------------------------------

WINDOW = 50                                                   # the vote's window, as the program


def voter(n, *near):
    """A voter array of n frames: nobody near the ball, except (first, stop, "H" / "A") stretches
    where that team's dot is near it."""
    out = np.full(n, NOBODY, dtype=np.int8)
    for first, stop, letter in near:
        out[first:stop] = HOME if letter == "H" else AWAY
    return out


def arrive(labels, votes, protected=None, blocks=None):
    protected = np.zeros(len(labels), dtype=bool) if protected is None else protected
    return control.change_on_arrival(labels, blind(len(labels), blocks), votes, protected,
                                     WINDOW)


@pytest.mark.parametrize("first_vote, expected", [
    (300 + 20, "H320 A280"),          # mid-pass: the change waits for the new team's first vote
    (300 + 1, "H301 A299"),
    (300, "H300 A300"),               # the new team near on the change's own frame: no move
    (300 + WINDOW, "H350 A250"),      # the last frame the vote is looked for
    (300 + WINDOW + 1, "H300 A300"),  # no vote of the new team within the window: no move
])
def test_a_change_mid_pass_moves_to_the_new_teams_first_near_vote(first_vote, expected):
    labels = spells(("H", 300), ("A", 300))
    votes = voter(600, (0, 280, "H"), (first_vote, 600, "A"))
    out, changed = arrive(labels, votes)
    assert text(out) == expected
    assert int(changed.sum()) == int(expected.split()[0][1:]) - 300


@pytest.mark.parametrize("length, first_vote, expected", [
    (40, 30, "H320 A20 H300"),        # at most half of the new spell (20 of 40)
    (41, 30, "H320 A21 H300"),        # an odd length: the shorter half moves
    (80, 30, "H330 A50 H300"),        # under half: the vote decides
    (2, 1, "H301 A1 H300"),           # even the shortest spell keeps a frame: none removed
])
def test_a_change_moves_at_most_half_of_the_new_spell(length, first_vote, expected):
    labels = spells(("H", 300), ("A", length), ("H", 300))
    votes = voter(len(labels), (300 + first_vote, 300 + length, "A"))
    out, _ = arrive(labels, votes)
    assert text(out) == expected


@pytest.mark.parametrize("parts", [
    (("H", 300), ("D", 50), ("A", 300)),          # after a dead ball: a restart, not a pass
    (("D", 50), ("A", 300)),                      # the first spell of the period
    (("H", 300), ("D", 1), ("A", 300)),
])
def test_a_spell_that_follows_a_dead_ball_is_not_touched(parts):
    labels = spells(*parts)
    start = len(labels) - 300
    votes = voter(len(labels), (start + 20, len(labels), "A"))
    out, changed = arrive(labels, votes)
    assert text(out) == text(labels) and not changed.any()


def test_d_x_and_protected_frames_never_change():
    labels = spells(("H", 300), ("A", 5), ("X", 5), ("A", 10), ("A", 280))
    labels[600 - 1] = "D"                         # a D elsewhere in the file
    protected = np.zeros(600, dtype=bool)
    protected[328:330] = True                     # two frames F2 wrote, just before the vote
    votes = voter(600, (330, 600, "A"))
    out, changed = arrive(labels, votes, protected)
    assert text(out) == "H305 X5 H18 A271 D1"
    assert changed.sum() == 5 + 18                # control: it did move the change
    assert not changed[(labels == "X") | (labels == "D") | protected].any()
    assert set(out[labels == "X"]) == {"X"} and out[-1] == "D"


def test_a_frame_f2_wrote_keeps_its_letter_even_where_that_splits_the_spell():
    # the definition's one exception to "no possession added": F2's frame starts the new spell
    labels = spells(("H", 300), ("A", 300))
    protected = np.zeros(600, dtype=bool)
    protected[300] = True
    out, changed = arrive(labels, voter(600, (320, 600, "A")), protected)
    assert text(out) == "H300 A1 H19 A280"
    assert changed.sum() == 19 and not changed[300]


def test_the_number_of_possessions_is_unchanged():
    labels = spells(("H", 300), ("A", 120), ("H", 40), ("D", 60), ("A", 200), ("H", 280))
    n = len(labels)
    votes = voter(n, (330, 420, "A"), (445, 460, "H"), (560, 760, "A"), (760, n, "H"))
    out, changed = arrive(labels, votes)
    assert changed.sum() == 30 + 20 + 40                      # control: three changes moved
    assert text(out) == "H330 A110 H20 D60 A240 H240"         # by 30, by half (20), by 40
    n_before = len([s for s in text(labels).split() if s[0] in "HA"])
    n_after = len([s for s in text(out).split() if s[0] in "HA"])
    assert n_before == n_after == 5
    assert ((out == "D") == (labels == "D")).all()            # no D added or removed


def test_a_change_is_read_inside_each_period():
    labels = spells(("H", 300), ("A", 300))
    votes = voter(600, (320, 600, "A"))
    out, _ = arrive(labels, votes, blocks=[(1, 0, 300), (2, 300, 600)])
    assert text(out) == "H300 A300"               # a new period, not a change of team
    out, _ = arrive(labels, votes)
    assert text(out) == "H320 A280"               # control: one period moves it


def test_change_on_arrival_leaves_its_inputs_unchanged():
    labels = spells(("H", 300), ("A", 300))
    votes = voter(600, (320, 600, "A"))
    protected = np.zeros(600, dtype=bool)
    before = (labels.copy(), votes.copy(), protected.copy())
    first, _ = arrive(labels, votes, protected)
    second, _ = arrive(labels, votes, protected)
    assert text(first) != text(labels)                        # control: it did write
    np.testing.assert_array_equal(first, second)
    for kept, now in zip(before, (labels, votes, protected)):
        np.testing.assert_array_equal(kept, now)
