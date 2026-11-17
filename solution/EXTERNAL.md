# External tools and data

Every outside tool and dataset this repository uses, as the organisers require (RULES.md:
anything in the pipeline must be accessible and explained). Licences are as declared by each
package's own metadata, read 2026-10-29.

**Where it is used:**
- **submission**: on the path that produces `Cataltina.csv`.
- **tests**: runs this repository's tests; nothing in the submission depends on it.
- **validation**: scoring our rules on other matches; not on the submission path and not in this
  repository.

## Code and libraries

| Tool | Version | Licence | Where it is used | Link |
|---|---|---|---|---|
| Organisers' repository: `games.py`, `starter/` (baseline, loaders), `evaluation/` (scorer, validator) | upstream `f8f2629` | MIT (repository contents) | **submission** (frame constants, baseline rule and smoothing, validation) | https://github.com/metrica-sports/more-than-a-hack-2027 |
| numpy | 2.4.6 | BSD-3-Clause (+ bundled MIT, Zlib, 0BSD, CC0) | **submission** | https://numpy.org |
| pandas | 3.0.6 | BSD-3-Clause | **submission** (via the scorer) | https://pandas.pydata.org |
| pytest | 9.2.0 | MIT | **tests** | https://pytest.org |
| kloppy | 3.18.0 | BSD | validation: reads the outside matches' tracking and events | https://github.com/PySport/kloppy |

## Data

| Dataset | What | Licence | Where it is used | Link |
|---|---|---|---|---|
| Competition match bundle (BAR-ATH) | tracking, Smart Tagging, homography, video | competition terms of data use | **submission**; never committed or redistributed | DATA.md |
| Public 5-minute ground truth | `evaluation/gt/` | MIT (in the organisers' repository) | correctness check only, never tuning (RULES.md) | organisers' repository |
| Metrica Sports sample data, games 1-3 | anonymised tracking and **human-annotated event data** of three other matches | **no licence file**; README asks users to credit the source. Used locally, never republished | validation: approximate practice answers derived from the events (our own script, not in this repository) | https://github.com/metrica-sports/sample-data (commit `e706dd5`) |
| DFL open matches (Bassek, Rein, Weber, Memmert 2025, "An integrated dataset of spatiotemporal and event data in elite soccer", *Scientific Data* 12:195) | seven matches of the 2022/23 Bundesliga and 2. Bundesliga: TRACAB tracking at 25 Hz (ball and players), event data, and on every frame **the league's own possession flag and ball-status flag** | **CC-BY 4.0**; data: DFL Deutsche Fussball Liga. Used locally, never republished | validation: measuring and checking rules on real tracking with real labels, as they are and with the ball taken out in stoppages to resemble our tracking. Read with kloppy. Not on the submission path | https://doi.org/10.6084/m9.figshare.28196177 |

**Training data:** none. No labels of any kind on the competition match. The practice answers above
are hand labels on **other** games, made by Metrica's annotators: event data for about 3 x 90
minutes (1,745, 1,935 and 3,620 events). Nothing is trained on them.

**How the numbers were chosen:** every number in the pipeline (distances, speeds, durations) was
measured and chosen on other matches: Metrica's three sample games and the seven DFL open
matches, which the organisers allow. None was chosen on the competition match, its video or its
public 5 minutes. Nothing is trained.

The DFL open matches are **other** matches too. Their possession and ball-status labels are the
league's (from its official match data), not Metrica's and not ours; a possession there is "the
player in control of the ball", which is close to the competition's definition but comes from
another provider, so scores on them are read as a second opinion, never as the competition's
truth. They are used to measure rules, choose their numbers and validate them, on real tracking
with real labels; nothing is trained on them, and no number in the pipeline is fitted to the
competition match.
