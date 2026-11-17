# Team Cataltina: More than a Hack 2027, possession

Our program labels every frame of the match's tactical video `H` (home has the ball), `A` (away
has the ball) or `D` (dead ball), from the tracking and the Smart Tagging of the competition
bundle. It is rules only: nothing is trained, and every number in it was measured on other
matches. It sits on top of the organisers' repository and imports their code (`games.py`,
`starter/`, `evaluation/possession_metrics.py`) without changing it.

## What it does, stage by stage

`solution/pipeline.py` runs three stages, in this order.

1. **The base: who is near the ball.** The organisers' baseline gives each frame to the team of
   the tracked player nearest the ball, then takes a majority vote over 2 seconds either side.
   We keep that vote but with fewer voters: the nearest player votes only when he is within 3 m
   of the ball, and nobody votes while the ball travels faster than 8 m/s (a hard pass or a lob
   flying past an opponent is not a change of team). Where nobody votes for a whole window, the
   ball stays with the team that had it.
2. **Dead ball** (`solution/dead_ball.py`), written over the base after the vote, never smoothed
   again: before each half's kickoff; outside the two halves; the ball wholly over a touchline or
   goal line (estimated from its last speed when the tracker lost it in the air); the restarts and
   goals marked in the Smart Tagging, dead from where the ball was lost to the kick that restarts
   play (with the players' slowdown deciding where a stoppage starts when the ball was simply
   lost); and a huddle of both teams standing still while the ball is not tracked. Then two
   corrections that never write over any of that: a ball lost right after a fast kick and seen
   again inside the pitch is in flight, so play is live (`X`, which the scorer reads as the team
   before it); and a ball the goalkeeper holds belongs to his team until he releases it.
3. **Possession** (`solution/control.py`), last:
   - in a stretch of more than 4 seconds where the tracker has lost the ball and that holds a
     dead ball, the team letters the vote carried into it become `D`;
   - a spell of one team shorter than 5 seconds, with the other team's spell before it and the
     other team's spell or a dead ball after it, becomes the other team's, unless its team
     touched the ball (the ball changed speed or direction by 4 m/s or more with one of its
     players within 1.5 m);
   - after a set piece (a corner, a goal kick or a free kick) whose Smart Tagging names a team,
     that team has the ball from the kick, whatever the vote says, until a player of the other
     team gets to the ball: until he touches it, or until he is the player near the ball (the
     nearest player within 3 m, the ball not faster than 8 m/s, and not the taker), whichever
     comes first;
   - a possession shorter than 1 second with a dead ball on both sides is dead;
   - last, a change of team is written where the new team first has the ball near (one of its
     players within 3 m, the ball not faster than 8 m/s), not in the middle of the pass that
     takes it there: the change moves later, by at most half the new possession.

Each module's docstring gives every rule in plain words, and its `# Params` block says where each
number comes from.

## Install

Python 3.11 or 3.13. From the repository root:

```
python -m venv .venv
.venv\Scripts\activate                       # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r solution/requirements.txt     # exact versions: the file must come out byte for byte
pip install -r solution/requirements-dev.txt # the tests only
```

## The data

The match data is licensed to registered participants only and is **never** in this repository.
Put the competition bundle in `data/` as the organisers describe (`data/README.md`, and `DATA.md`
in the organisers' repository, https://github.com/metrica-sports/more-than-a-hack-2027).
`data/` and `out/` are in `.gitignore`. The data's licence is the organisers' (their RULES.md,
"The data"): never commit it or pass it on. Install the safety hook, which refuses any commit
containing match data:

```
cp solution/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit               # macOS / Linux only
```

## The one command

```
python solution/pipeline.py                  # -> out/submission/Cataltina.csv
```

It prints label counts and the file's SHA-256, which must be:

```
1692adb766f161c4fe17ffcbd71c8b3363a10f2d937d43bae9be22685e4cfe0f
```

The program has no randomness and no timestamps, and writes `\n` line endings on every system,
so the same data gives the same bytes. We checked it on Windows: two builds on Python 3.13 and
one in a clean Python 3.11 environment installed from `solution/requirements.txt` give this
SHA-256.

## Check it

```
python evaluation/validate_submission.py out/submission/Cataltina.csv   # the organisers' validator
python solution/check_seals.py out/submission/Cataltina.csv             # rows, frame order, labels,
                                                                        # the validator, the SHA-256
python solution/check_seals.py --rebuild     # builds the file twice in separate processes;
                                             # both SHA-256 must equal the file's
python -m pytest solution/tests -q           # the tests, on small invented files only
```

`check_seals.py --rebuild --python <other python>` makes the second build under another
interpreter, for example a clean Python 3.11 environment. `rebuild_baseline.py` rebuilds the
organisers' four baselines and checks that our array version of their nearest-player rule gives
their answer on every frame.

## How the numbers were chosen

Every number in the program (distances, speeds, durations) was measured and chosen on other
matches: Metrica's three public sample games, with approximate answers made from their event data,
and the seven DFL open matches, with the league's own possession labels. Both are listed in
[EXTERNAL.md](EXTERNAL.md), which the organisers allow. None was chosen on the competition match,
its video or its public 5 minutes, and nothing is trained. The public 5 minutes were used only to
check that frames, labels and the scorer line up.

## Licences

`games.py`, `starter/` and `evaluation/` are the organisers' code, under the MIT licence in
[LICENSE](../LICENSE) (Metrica Sports), which covers their repository's contents but not the
match data. Outside tools and datasets, and their licences, are in [EXTERNAL.md](EXTERNAL.md).
