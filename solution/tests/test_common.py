import pytest

import common
from common import PRE_KICKOFF, graded_blocks, kickoff_frames, match_clock, period_of, video_time

BLOCKS = graded_blocks()
KICKOFFS = kickoff_frames()


def test_graded_blocks_match_data_md():
    # DATA.md "Which frames are graded": 11,185..90,651 and 110,808..190,197, inclusive.
    assert BLOCKS == [(1, 11185, 90652), (2, 110808, 190198)]


def test_kickoffs_match_faq():
    assert KICKOFFS == {1: 11185, 2: 113742}


@pytest.mark.parametrize("frame, expected", [
    (0, "0:00:00"),
    (24, "0:00:00"),
    (25, "0:00:01"),
    (25 * 3661, "1:01:01"),
])
def test_video_time(frame, expected):
    assert video_time(frame) == expected


@pytest.mark.parametrize("frame, expected", [
    (0, ""),                          # before the match
    (11185, "00:00"),                 # 1st half kickoff
    (11185 + 25 * 61 + 24, "01:01"),  # last frame of 01:01
    (90651, "52:58"),                 # last graded frame of the 1st half
    (90652, ""),                      # halftime
    (110808, PRE_KICKOFF),            # graded, but the 2nd half has not kicked off
    (113741, PRE_KICKOFF),
    (113742, "45:00"),                # 2nd half kickoff
    (190198, ""),                     # after the final whistle
])
def test_match_clock(frame, expected):
    assert match_clock(frame, BLOCKS, KICKOFFS) == expected


@pytest.mark.parametrize("frame, expected", [
    (11184, "before the match"),
    (11185, "1st half"),
    (90652, "halftime"),
    (110808, "2nd half"),
    (190198, "after the final whistle"),
])
def test_period_of(frame, expected):
    assert period_of(frame, BLOCKS) == expected


def test_write_labels_uses_lf_on_every_os(tmp_path):
    path = common.write_labels(str(tmp_path / "x.csv"), list("HADX"))
    with open(path, "rb") as handle:
        content = handle.read()
    assert b"\r" not in content
    assert content == b"frame,label\n0,H\n1,A\n2,D\n3,X\n"


def test_sha256_newline_normalising_matches_crlf_copy(tmp_path):
    lf = tmp_path / "lf.csv"
    crlf = tmp_path / "crlf.csv"
    lf.write_bytes(b"frame,label\n0,H\n")
    crlf.write_bytes(b"frame,label\r\n0,H\r\n")
    assert common.sha256_of(str(lf)) != common.sha256_of(str(crlf))
    assert common.sha256_of(str(lf)) == common.sha256_of(str(crlf), normalise_newlines=True)
