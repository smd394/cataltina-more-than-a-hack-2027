"""Synthetic stand-ins for the match files, in the formats DATA.md documents.

Every number here is invented. No test reads data/, out/ or any file the organisers ship about
the match; a test that needs a ground truth or a data bundle writes its own into tmp_path.
"""

import os
import sys

import pytest

SOLUTION = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SOLUTION not in sys.path:
    sys.path.insert(0, SOLUTION)

import common  # noqa: E402,F401  (puts the repo root and evaluation/ on the path)
from games import asset_paths  # noqa: E402

HOME_CLUB, AWAY_CLUB = "Rivertown FC", "Hillside Athletic Club"

META_XML = """<?xml version="1.0" encoding="utf-8"?>
<main><Metadata>
  <Teams>
    <Team id="T1"><Name>{home}</Name></Team>
    <Team id="T2"><Name>{away}</Name></Team>
  </Teams>
  <Players>
    <Player id="P1" teamId="T1"><Name>Track 1</Name></Player>
    <Player id="P2" teamId="T2"><Name>Track 2</Name></Player>
  </Players>
</Metadata></main>
""".format(home=HOME_CLUB, away=AWAY_CLUB)


def atd_line(frame_id, ball_visible):
    """One raw ATD record: two player slots and the ball, NaN where not observed."""
    ball = "0.5,0.5,0.0" if ball_visible else "NaN,NaN,NaN"
    return "{}:0.40,0.50,1.0;NaN,NaN,NaN:{}\n".format(frame_id, ball)


def smart_tagging_xml(instances):
    """SportsCode XML from [(start_s, end_s, code, {group: text})]."""
    body = []
    for number, (start, end, code, tags) in enumerate(instances, start=1):
        labels = "".join("<label><group>{}</group><text>{}</text></label>".format(g, t)
                         for g, t in tags.items())
        body.append("<instance><ID>{}</ID><start>{}</start><end>{}</end><code>{}</code>{}"
                    "</instance>".format(number, start, end, code, labels))
    return ("<?xml version=\"1.0\" encoding=\"utf-8\"?><file><ALL_INSTANCES>{}"
            "</ALL_INSTANCES></file>".format("".join(body)))


@pytest.fixture
def data_dir(tmp_path):
    """A tiny synthetic tactical bundle at the paths games.asset_paths() expects."""
    root = tmp_path / "bundle"
    paths = asset_paths(data_dir=str(root))
    os.makedirs(os.path.dirname(paths["atd_raw"]))
    with open(paths["atd_meta"], "w", encoding="utf-8") as handle:
        handle.write(META_XML)
    with open(paths["atd_raw"], "w") as handle:
        handle.writelines(atd_line(frame_id, frame_id % 2 == 1) for frame_id in range(1, 101))
    with open(paths["smart_tagging"], "w", encoding="utf-8") as handle:
        handle.write(smart_tagging_xml([
            (2.0, 4.0, "BUILD UP", {"Team": "Rivertown", "Half": "1st Half"}),
            (3.0, 3.5, "SET PIECES", {"Team": "N/A", "Half": "1st Half", "Type": "Throw-in"}),
            (5.0, 9.0, "PROGRESSION", {"Team": "Hilside Athletic", "Half": "1st Half"}),
        ]))
    return str(root)


def write_csv(path, labels, frames=None):
    """A frame,label CSV written plainly, so tests can also write broken ones."""
    frames = range(len(labels)) if frames is None else frames
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(f, l) for f, l in zip(frames, labels))
    return str(path)
