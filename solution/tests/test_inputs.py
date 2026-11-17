import pytest

import inputs
from conftest import AWAY_CLUB, HOME_CLUB, atd_line


def test_ball_seen_maps_atd_frame_id_onto_zero_based_video_frame(tmp_path):
    # THE alignment guard: ATD frame_id 1 is video frame 0 (DATA.md "Frame alignment").
    raw = tmp_path / "atd.txt"
    raw.write_text(atd_line(1, True) + atd_line(2, False) + atd_line(3, True))
    seen = inputs.ball_seen(str(raw), n_frames=4)
    assert seen.tolist() == [True, False, True, False]


def test_ball_seen_skips_malformed_lines(tmp_path):
    raw = tmp_path / "atd.txt"
    raw.write_text("not a record\n" + atd_line(2, True))
    assert inputs.ball_seen(str(raw), n_frames=3).tolist() == [False, True, False]


@pytest.mark.parametrize("tag, expected", [
    ("Rivertown", "H"),
    ("Hilside Athletic", "A"),       # misspelt, as the real tags misspell a club
    ("Hillside Athletic Club", "A"),
    ("N/A", "N/A"),
    ("", "N/A"),
    (None, "N/A"),
    ("Zzzz", "N/A"),                 # resembles neither club
])
def test_side_of_team_tag(tag, expected):
    assert inputs.side_of_team_tag(tag, HOME_CLUB, AWAY_CLUB) == expected


def test_club_names_home_is_first_team(data_dir):
    assert inputs.club_names(data_dir=data_dir) == (HOME_CLUB, AWAY_CLUB)


def test_tag_rows_convert_seconds_to_frames_and_sides(data_dir):
    rows = inputs.tag_rows(inputs.load_tags(data_dir=data_dir), HOME_CLUB, AWAY_CLUB)
    assert [(r["Code"], r["Side"], r["First frame"], r["Last frame"]) for r in rows] == [
        ("BUILD UP", "H", 50, 99),       # 2.0 s .. 4.0 s at 25 fps, end exclusive
        ("SET PIECES", "N/A", 75, 87),   # round(3.5 * 25) = 88 -> last frame 87
        ("PROGRESSION", "A", 125, 224),
    ]
    assert rows[0]["Starts at (video)"] == "0:00:02"


def test_tags_by_second_lists_every_second_a_tag_touches(data_dir):
    rows = inputs.tag_rows(inputs.load_tags(data_dir=data_dir), HOME_CLUB, AWAY_CLUB)
    active = inputs.tags_by_second(rows, n_seconds=10)
    assert active[1] == ""
    assert active[2] == "BUILD UP (H)"
    assert active[3] == "BUILD UP (H) | SET PIECES (N/A)"
    assert active[4] == ""
    assert active[8] == "PROGRESSION (A)"
    assert active[9] == ""
