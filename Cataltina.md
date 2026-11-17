# Possession by rules, from the tracking

**Team Cataltina** · More than a Hack 2027, Round 1 · technical document

## 1. Summary

We label every frame of the tactical video `H`, `A` or `D` with rules only. Nothing is trained. The
program starts from the organisers' baseline (the team of the nearest tracked player, then a majority
vote over 2 seconds either side) and adds three stages. **The base** keeps that vote but lets a player
vote only when he is near the ball and the ball is not flying past. **The dead-ball rules** write `D`
where there is evidence that play stopped: the ball off the pitch, a tagged restart, a goal, a huddle.
**The possession rules** undo short false spells, give a tagged restart to its team, and write each
change of team where the new team first has the ball near, not in the middle of the pass that takes it
there.

How each step scored. spellF1 is the organisers' ranking metric: predicted and true possessions
(maximal runs of `H` or of `A`) are matched one to one by temporal overlap, and the F1 of that matching
is averaged over overlap thresholds 0.3 to 0.7. These are **scores on other matches**, with the
organisers' scorer, against the answers described in section 4. **The competition match was never
scored to choose anything.**

| Program | Metrica sample games: mean spellF1 (game 1 / 2 / 3) | Our match: possessions (mean length) |
|---|---|---|
| Organisers' baseline `smooth50`, rebuilt frame for frame | 0.4345 (0.4584 / 0.4426 / 0.4026) | — |
| + the dead-ball rules | 0.5034 (0.5172 / 0.5116 / 0.4815) | 540 (7.0 s) |
| + the possession rules | 0.6280 (0.6354 / 0.6532 / 0.5953) | 284 (12.3 s) |
| + the base "near the ball and not in flight" | 0.6679 (0.6561 / 0.7007 / 0.6470) | 249 (14.0 s) |
| + the change on arrival | 0.7105 (0.7018 / 0.7338 / 0.6958) | 249 (14.0 s) |
| + the restart rule's second end (**the submitted program**) | **0.7105** (the same: these games have no tags) | **257** (13.5 s) |

On the seven DFL open matches (section 4), mean spellF1 against the league's own labels:

| DFL matches | Before the base | With the base | The change on arrival adds | Final program |
|---|---|---|---|---|
| as they are | 0.4363 | 0.4418 | +0.0598 | 0.5016 |
| made to look like ours (bench 1) | 0.4646 | 0.4979 | +0.0656 | 0.5635 |
| bench 2 (no player positions dropped) | 0.5008 | 0.5240 | +0.0428 (mean of four levels) | — |
| bench 3 (no ball lost in play) | 0.4671 | 0.4993 | +0.0403 (mean of four levels) | — |

The "final program" column is the sum of the two recorded figures before it (±0.0001 from rounding).
The last step, the restart rule's second end, reads the Smart Tagging, which no outside set has, so it
moves neither table; its own test is in section 5.
The submitted file holds H 51892, A 35102, D 112325, X 0 frames and 257 possessions in the graded halves.

## 2. The approach and why

**Rules before models.** The answer key is mechanical: a frame is dead where the corrected tracking
marks the ball dead, and otherwise belongs to the team the tracking names as owning it (CHALLENGE.md).
No labels of this match may be used, and the public window holds 17 possessions, too few to train or
tune anything. A rule can be written in plain words, read back by someone who knows football, and
measured on other matches. We kept a model for the place rules stopped helping, and did not reach it.

**The tracking and the players are the base; the Smart Tagging confirms.** The tags miss restarts (the
whole match holds 5 tagged free kicks, DATA.md) and are loose in time: a set-piece tag usually starts
while the ball is still dead. A tag is a pointer; the moment itself (the ball's loss, the kick that
restarts play) is found in the tracking.

**General football logic, with numbers from the Laws or from other matches.** The Laws give some
numbers (the ball's radius, the penalty area); the rest were measured as physical quantities on
Metrica's three sample games and seven Bundesliga matches. No rule names a frame of our match, and no
number was read off our match, its video or its public 5 minutes. Counts on our match only checked that
a rule changes some letters and leaves the other rules' frames alone. Two choices leaned on counts
from our own tracking: the wider reach of the touch test, on the share of frames with a player near the
ball, and the fallback for a goal's kick-off, kept because it changes a letter. Neither used a score or
an answer key.

**Built in steps, each measured before it went in, against a pass mark fixed before the scores were
read.** For the first two stages the mark was "the mean on the sample games must not fall". For the
base, the change on arrival and the restart rule's second end it was stricter: raise the mean on the sample games and on the
Bundesliga copies made to look like our tracking, and be below the program in no cell, a cell being one
outside set at one level of degradation scored against one version of its answers (section 4). Where a
candidate was defined after a table had been read, we say so.

## 3. The possession rules

Three stages, in this order. Smoothing happens only in the first, so a dead-ball stretch keeps its exact
first and last frame (edges matter: a perfect answer 1 s late scores 0.6539).

### 3.1 The base: "near the ball and not in flight"

The organisers' baseline gives each frame to the team of the tracked player nearest the ball and takes
a majority vote over ±50 frames (2 s); with no ball for that long it writes `D`. We rebuilt it and check
that it gives their `smooth50` file on every frame. Our base keeps the vote with fewer voters:

- the nearest player votes for his team only when he is **within 3 m** of the ball;
- **nobody votes while the ball travels faster than 8 m/s**: a hard pass or a lob that flies past an
  opponent is not a change of team;
- where nobody votes for a whole window, **the ball stays with the team that had it**. The base writes
  no new `D`.

The 3 m and 8 m/s were chosen among 2, 3, 5 and 8 m and 8, 12 and 16 m/s on the sample games and the
Bundesliga matches. Every neighbouring setting (2.5 to 3.5 m with 6 to 10 m/s) gains about the same on
the mean (+0.021 to +0.024).

### 3.2 The dead-ball rules

They write `D` over the base, in this order:

- **Before kick-off and outside the halves.** Every graded frame before a half's kick-off is `D`; so is
  every frame outside the two halves (never graded).
- **Ball off the pitch.** The ball is out when it is wholly over a touchline or goal line (Law 9): its
  centre past the line by at least its radius, 11 cm (Law 2). It is `D` until the ball is next tracked
  inside. A ball back inside within 1 s was never out (nobody restarts play that fast), and two out
  stretches less than 1 s apart are one stoppage.
- **Out while the tracker had lost it.** The tracking has no height, so a ball in the air is often lost.
  If the ball is next seen off the pitch, the crossing is estimated by carrying it on at its last
  tracked speed and direction; if that is impossible, the stoppage starts where the ball was lost.
- **Restarts from the Smart Tagging.** The stoppage runs from where the ball was lost or went out after
  the last kick before a set-piece tag, to the restart: the first kick inside the tag (the ball inside
  the pitch at 4 m/s or more, still that fast 0.2 s later). A later kick replaces it only if the
  players react to it within 1.6 s (a player within 15 m speeds up by 0.5 m/s over 2 s and closes on
  the ball, or a far player of each team speeds up). On the sample games this test passes 91.7% of 240
  restarts and 27.8% of 1,150 dead moments; 1.6 s is the 90th percentile of real restarts that fail it
  at first. A tag with no ball loss before it, or no kick inside it, writes nothing: a tag is evidence,
  not an answer. Tag types are treated alike, since a type can be wrong.
- **Where a stoppage starts when the players were still running.** If the stoppage starts at a lost
  ball while both teams are still clearly running (the slower team's mean speed over 4 s at least
  1.54 m/s, measured on the sample games), it starts where they slow down below that, less 1 s.
- **Goals.** `D` from the ball wholly over the goal line (after the goal tag starts) until the kick-off
  after the goal. When no kick-off kick can be seen, the start of the kick-off tag is the restart.
- **A huddle.** While the ball is not tracked: at least 6 players, 2 or more of each team, within 5 m of
  one player, moving at 1 m/s or less on average, for at least 1 s. A crowded box at a corner is not a
  huddle: players move fast there.

Then two corrections for a lost ball, which never write over any of the above:

- **A ball in flight** lost within 1 s of a kick, still at kick speed (4 m/s), and seen again inside the
  pitch within 4 s, is live: those frames become `X`, which the scorer reads as the team before.
- **The goalkeeper's ball.** The keeper is his team's player nearest its own goal. When the ball is lost
  within 3 m of him inside his own penalty area (Law 1: 16.5 m by 40.32 m) and seen again inside the
  pitch moving away from him within 17.7 s, those frames are his team's. 13.7 s is the 90th percentile
  of release times over 66 catches on the sample games; 4 s more is the longest a kicked ball stays up.
  The rule writes nothing in a half where the two teams' goals do not come out opposite.

### 3.3 The possession rules

They come last:

- **Long ball gaps.** In a gap where the tracker loses the ball for more than 4 s (longer than the vote
  can bridge) and that holds a dead ball, every team letter the vote carried into it becomes `D`. On the
  sample games four in five of those letters are dead ball.
- **A short spell is not a possession unless its team touched the ball.** A spell of one team shorter
  than 5 s, with the other team's spell before it and the other team or a dead ball after it, becomes
  the other team's, unless its team touched the ball: the ball changed speed or direction by 4 m/s or
  more with one of that team's players within 1.5 m. The shortest spell is read first, then the rest
  again. The 5 s is the longest length at which the rule, with the touch test seeing nothing, is still
  not below having no rule on the sample games.
- **A tagged restart's team has the ball.** At a set-piece tag that names a team (a corner, a goal
  kick, a free kick), where the dead-ball rule found the stoppage, the tag's team has the ball from the
  kick, whatever the vote says, until a player of the other team gets to the ball: until he touches it,
  or until he is the player near the ball (the nearest player within 3 m, the ball not faster than
  8 m/s), whichever comes first. Neither is counted in the first 0.4 s; the taker's dot is excluded,
  whatever colour it shows; the rule stops at the next dead ball. It has no number of its own.
- **Dead islands.** A possession shorter than 1 s with a dead ball on both sides is dead (the same
  1 s as for the ball off the pitch).
- **The change on arrival.** "A change of team is written where the new team first has the ball near.
  The vote changes team in the middle of a pass; the possession changes when the new team gets the
  ball." Where one team's spell follows the other's straight away, the change moves to the first frame,
  within the vote's window, where a player of the new team is within 3 m of the ball and the ball is not
  faster than 8 m/s; never by more than half the new spell. It moves a change later, never earlier, and
  writes no `D`. A spell that follows a dead ball is a restart and is not touched. It has no number of
  its own: it reuses the base's.

No possession rule changes a `D` written by the dead-ball rules, an `X`, or a frame the goalkeeper rule
wrote, and none joins spells across a dead ball or across half-time.

### 3.4 Where we drew the line

A **loose or contested ball** stays with the team that had it (nobody within 3 m, nobody votes). A
**challenge, a touch without control or a deflection** that gives the other team a spell under 5 s is
undone unless that team touched the ball; the rule cannot tell a controlled touch from a deflection, so a
spell with a touch is kept. A **flick** is undone by the same rule, and a possession under 1 s between
two dead balls is dead. A **ball in the air** (faster than 8 m/s) has no voter, and the change is written
where it arrives. The **goalkeeper's ball** is his team's while he holds it, up to 17.7 s. A **restart**
is dead until the kick, then belongs to the team the tag names until the other team touches the ball or
has a player near it. A
**goal** is dead from the ball over the line to the kick-off.

## 4. Data and assets

**Used, from the competition bundle:** the tactical recording's tracking (ball and players, 25 fps,
row i = video frame i) and the Smart Tagging (set-piece, goal and kick-off tags). Nothing else is on the
path that writes the file; the broadcast recording was not used.

**Used to understand the task and design the rules, never as input:** the tactical video and the public 5 minutes. The public window
was scored only to check that frames, labels and the scorer line up. Our teammates watched it with the
organisers' correct answers and the tags and wrote what they saw in plain words; those words became
general rules (for example: "the official answer switches at the new team's first touch when that
touch plays the ball", never while the ball is still travelling to them), never labels, frames or numbers. Three
situations seen in the video where the tracker loses the ball (a ball in the air, a goalkeeper holding
the ball, a huddle) became general rules whose numbers come from the Laws and the sample games.

**Tried or set up, and dropped:**

- **The homography** served only to draw the tracking over the video for our own understanding. The
  rules work in pitch metres from the tracking, so it is not on the path.
- **Finding the ball ourselves in the video** (a ready-made detector, kept in a separate environment)
  was planned for the case where the rules stopped improving. They kept improving, and a detector on a
  GPU is not bit-stable, which would break the byte-for-byte rebuild. Not built.
- **The phase tags as a possession rule** and **cleaning wrongly coloured dots**: built, measured and
  taken out (section 5).

**Training data: none.** Nothing is trained. No labels of any kind on the competition match were made or
used. Every number was measured and chosen on two outside sets, which the organisers allow:

| Set | Source and licence | Labels: who made them, how much | How we used it |
|---|---|---|---|
| Metrica sample games 1-3 | Metrica Sports public sample data (no licence file; credit asked; used locally, never republished) | Tracking plus event data by Metrica's annotators: 1,745 / 1,935 / 3,620 events. We made approximate frame answers from the events ourselves, by a fixed rule (possession from each on-ball action; dead from a ball out, a foul, an offside, to the restart) | measuring numbers; scoring every step |
| DFL open matches, seven | Bassek, Rein, Weber, Memmert 2025, *Scientific Data* 12:195; data by DFL Deutsche Fussball Liga; **CC-BY 4.0** | TRACAB tracking at 25 Hz and, on every frame, the league's own possession and ball-in-play flags, from its official match data | measuring and checking rules on real tracking with real labels |

**Copies made to look like our tracking.** Our match shows 5.6 touches per tracked minute (a sharp
change of the ball's speed with a player at it), a quarter of the DFL matches' 21.0 and a sixth of the
sample games' 32.5, and our ball is lost mostly in stoppages, while the DFL ball is tracked standing still
in dead time. So every rule was also scored where touches are as scarce as ours (one touch in four or in
eight seen, or the ball smoothed over 0.8 s), and on DFL copies with the ball removed in 90% of stoppages
from their start, 7% of player positions dropped and 2% of play with the ball lost ("bench 1"; bench 2
drops no positions, bench 3 loses no ball in play). Because no outside set can tell how many very short
possessions the competition's answer key holds, each table was also read against the outside answers
with one team's spells under 1, 2, 3 or 5 s taken out. On the submission path: Python, numpy and pandas
(pinned) and the organisers' loaders, scorer and validator, unchanged; kloppy read the outside sets.

## 5. What we explored and left out

Each was measured on other matches; the figures are mean spellF1 changes against the program of the
time.

- **Cleaning the dots** (ghost dots, the goalkeeper's colour): built, changed no
  letter on our match, and could move a real attacker on a tracking with missing dots. Taken out. A
  check that a player "moves with his teammates" was at chance level on clean games (48.3-51.7%).
- **Live or dead from a speed level of the players:** practice 0.5011 → 0.4998; its live calls landed
  on real stoppages 88% of the time, mostly in a stoppage's first ~10 s. Not wired. **"Dead until the
  ball is played again":** failed the practice floor twice (0.5010 and 0.5023 against 0.5034).
- **The organisers' phase tags as a possession rule:** a veto that keeps a spell its own team's phase
  backs gains with exact phases and falls below the floor once a phase spans a possession of the other
  team of 6 s or more, which real phases do. Taken out; the tags stay a witness.
- **A shorter vote (±1 s):** +0.030 on the sample games; the sign turns on real tracking with gaps.
  **A touch test that sets its own thresholds from the match it reads:** -0.001 to -0.012. **A ball
  that stands still is dead:** no outside set has a ball tracked in dead time, so it cannot be measured.
- **Possession read from touches instead of the nearest player:** +0.10 to +0.14 with every touch seen,
  a loss of 0.07 to 0.20 once touches are as scarce as ours. A touch counted for 2 s only: a tie.
- **The short-spell length at 4 s instead of 5 s** (and later 3 s): better against answers with many
  short possessions, worse against answers with few. On the final day 3 s and 4 s failed the "never
  below" line again, also on top of the change on arrival (4 s there: 39 of 40 cells, smallest -0.0023).
- **A change of team moved to the exact touch:** +0.009 to +0.014 with every touch seen, about zero on a
  tracking like ours. **Throw-ins given to the team of the player at the ball:** +0.001 to +0.008 with
  every touch seen, -0.004 where touches are scarce.
- **"Nobody near the ball = dead ball":** its gain came from the DFL ball standing still in dead time;
  on our match it would have cut 21 possessions in two.
- **Live or dead from the players' movement alone, the ball hidden** (the DFL matches): a "live" call
  is right on 90 to 98% of frames, but 44 to 46% of dead frames are called live in a stoppage's first
  2 s. No sign was "never below" (-0.008 to 0 where every lost-ball stretch is dead, about -0.006 to
  +0.009 on the copies like ours). A small logistic regression on players' speeds, trained on those
  matches only, changed the result little; even the league's own ball-in-play flag as the sign gains
  only +0.008 to +0.023 there. One variant, "a whole gap is live only if the players never stop during
  it" (-0.0005 where every gap is dead, -0.002 on the sample games, +0.006 to +0.009 at one level like
  ours), was **defined after the scores were read** and stays unbuilt.

**The final day.** Ideas from looking at our labels beside the tags (no video), all measured, none
built: keeping the team through a lost ball while an attacking phase tag names it (the tags cover 48.7%
of the time our rules know is dead; at a 10 s run-on, -0.032 where every gap is dead); a restart only
from where the Laws put the ball (at 8 of 12 corners the kick taken as the restart starts more than
8.5 m from every corner, but on the one outside set that can judge it, -0.0020 to +0.0001); the goal
stoppage (no kick-off taken away from the centre; at most about 48 frames at stake); the other team's
first touch after a restart counted only from where the Laws allow opponents (0 frames change on our
match; -0.0011 to +0.0004 outside); a flicker of 1.5 to 5 s between two dead balls made dead (below the
program in every cell: of what it would make dead, the outside answers call dead only 2 to 9% on the
sample games and 0 to 3% on bench 1); keeping the team through a lost ball when the same team has it on
both sides (-0.0463 on the pass mark; -0.028 to -0.042 when restart tags are missed, since a foul after
which the same team restarts looks the same); two or more attacking tags of one team at once (they cover
31.6% of the time our rules know is dead); the players' signals for a lost ball with direction, line and
marking added (right in a stoppage's first 2 s only 55 to 57% of the time); the keeper's catch read from
both teams turning away from goal (86% of what it would write is dead in the outside answers); "stay
dead until play is confirmed".

**How the first rule of that day was found.** We audited our own measuring setup: the scorer, the
alignment of the outside answers, the caches. It held, and our dead share (45.2% of graded frames) sits
inside the real dead share of the DFL matches (40.7 to 51.7%). But it showed that on every outside set
our changes of team came a few frames early: every set scored higher with our labels moved later. That
became the change on arrival. On the pass mark it gained in 8 of 8 cells (mean +0.0357), was below in
none of 40 (smallest +0.0099), raised every sample game and every match of bench 1 at every level, and
raised the practice mean from 0.6679 to 0.7105. On our match it moves 2,860 letters and left the
possessions at 249.

**A second look, and the last rule.** The ideas of the final day were then measured again, each in
every form that made football sense: 130 forms written down before any was scored, 12 more added after
and marked so. Six ideas stayed out. No way of keeping a team through a lost ball is free of loss where
every gap is dead. No rule on short possessions passes: after the change on arrival most of them are
real. "Stay dead until a kick" gains only where the ball is shown again during a stoppage (+0.003 to
+0.005 on a copy built for that) and costs a little everywhere else. A restart judged by where its
taker stands changes nothing measurable. A goalkeeper's catch can be told from a ball going out, but
too rarely to pass. A short-spell length of 4 or 4.5 s still fails against answers with few short
possessions. One form passed: **the restart rule's second end** (section 3.3: the spell also ends when
the other team has a player near the ball). Of sixteen forms of that rule it alone passed, scored
against the rule ending at the touch only, with restarts made from the outside matches' own events: up
in 8 of 8 cells (mean +0.0052), below in none of 40 (smallest +0.0008), the sample games with restarts
0.7154 to 0.7167, and up on 32 of 32 rows when the program finds the restarts itself from tags made
from those events. Scored again as built, it gave the same figures. Where touches are scarce the old
rule ran on into the other team's possession; this one stops. On our match it changes 689 letters
beside 6 restarts, and the possessions go from 249 to 257.

## 6. Limitations

- **The ball is missing on 40.7% of graded frames**, and those stretches are mostly written as dead
  ball. This is our largest known weakness: where the tracker lost the ball in live play, we cut a
  possession in two. Every attempt to read live play from the players alone failed on other matches.
- **The outside answers are a second opinion, not the competition's truth.** The sample-game answers
  are ours, made from event lists; the DFL labels come from another provider. On the sample games the
  tagged-restart rule is tested in its best case (its restarts and the answers come from the same
  events).
- **How many very short possessions the answer key holds cannot be measured on any other match**, so
  the 5 s length stays: 4 s scores better against answers with many short possessions and worse
  against answers with few. The touch test fires about half as often on our match as on the sample games
  (short spells kept 20% against 47-49%), so whether our remaining long spells without a touch are
  real is open. "Never worse than no rule" holds on the mean of the three sample games, not on each.
- **The base's gain is about the same at every neighbouring setting**, but its "never below" mark rests
  on one of the seven DFL matches: without that match the built setting misses two cells by 0.002.
- **The ball-in-flight rule has never fired** on any match we hold. **A wrongly coloured dot near the
  ball cannot be detected**; it votes for the wrong team. **No bootstrap over the outside tables:** the
  gains are point figures on 3 + 7 matches.
- **The change on arrival was found after earlier tables had been read.** It was then scored once, as
  built, against a pass mark fixed beforehand, and had no separate review session. In one rare case (a
  frame the goalkeeper rule wrote at the very start of a possession) it can add a one-frame possession:
  once on the sample games, never on our match.
- **The restart rule's second end is a small gain** (+0.005 on the mean), found on the final day and
  also without a separate review session. On a second draw of the copies' degradations one of 40 cells
  read -0.0001. It ends the restart team's spell when a player in the other team's colour is nearest
  the ball; a wrongly coloured teammate does that too, and no outside set has such dots.
- **With more time** we would detect the ball in the long gaps from the video, deterministically on
  CPU; test a stoppage sign on an outside set whose ball is lost in live play as ours is; and label a
  few other matches by hand to measure how many short possessions a tracking-derived answer holds.

## 7. Reproduction

With the competition bundle in `data/`, Python 3.11 or 3.13 and the pinned requirements:

```
pip install -r solution/requirements.txt
python solution/pipeline.py          # -> out/submission/Cataltina.csv
```

The file's SHA-256 must be `1692adb766f161c4fe17ffcbd71c8b3363a10f2d937d43bae9be22685e4cfe0f`. No
randomness, no timestamps, `\n` line endings: two builds on Python 3.13 and one in a clean Python 3.11
environment give this SHA-256. Checks: the organisers' `validate_submission.py`; `check_seals.py
--rebuild` (two builds in separate processes); `pytest solution/tests` (small invented files only).
Details are in the repository's `solution/README.md`.

## 8. The public 5 minutes

A correctness check only; its score never chose anything: spellF1 0.4727 [0.32-0.63] with 1,000
bootstrap draws, macroF1 0.7004, 16 possessions against the answer's 17.
