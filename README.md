# Team Cataltina: More than a Hack 2027

Team Cataltina's submission to More than a Hack 2027 (the possession task): a rules-only program
that labels every frame of the match's tactical video with the team in possession or a dead ball.

- **How to rebuild the submission file:** [solution/README.md](solution/README.md) (install, the
  one command, the SHA-256 it must give, the checks).
- **The method:** [Cataltina.md](Cataltina.md), the technical document.
- **Outside tools and data, and their licences:** [solution/EXTERNAL.md](solution/EXTERNAL.md).

It is built on the organisers' repository,
https://github.com/metrica-sports/more-than-a-hack-2027, whose code and public files (`games.py`,
`starter/`, `evaluation/`, `data/README.md`) are included unchanged under their
[LICENSE](LICENSE). Their documents (the challenge, the data, the evaluation) are in their
repository.

The match data is not here: it is licensed to registered participants only. Put the bundle in
`data/` as `data/README.md` describes; `.gitignore` keeps it out of every commit.
