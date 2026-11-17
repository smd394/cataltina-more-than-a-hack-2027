"""The pre-commit hook, exercised with real `git commit`s in a throw-away repository."""

import os
import shutil
import subprocess

import pytest

HOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks",
                    "pre-commit")
GIT = shutil.which("git")
MATCH = "BAR" + "-ATH_"  # built at runtime: the match bundle's file-name prefix

pytestmark = pytest.mark.skipif(GIT is None, reason="git is not installed")


def git(repo, *args):
    return subprocess.run([GIT, "-c", "user.name=test", "-c", "user.email=test@example.com",
                           "-c", "core.autocrlf=false"] + list(args), cwd=repo,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


@pytest.fixture
def repo(tmp_path):
    path = str(tmp_path / "repo")
    os.makedirs(path)
    git(path, "init", "-q")
    hook = os.path.join(path, ".git", "hooks", "pre-commit")
    shutil.copyfile(HOOK, hook)
    os.chmod(hook, 0o755)
    return path


def stage(repo, relative, text="x\n"):
    full = os.path.join(repo, *relative.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", newline="\n") as handle:
        handle.write(text)
    assert git(repo, "add", "-f", relative).returncode == 0


@pytest.mark.parametrize("path, refused", [
    ("data/probe.txt", True),
    ("Data/Probe.txt", True),                                     # case does not matter
    ("data/tactical/x.txt", True),
    ("data/README.md", False),                                    # the organisers' readme
    ("out/tables/summary.csv", True),
    ("external/metrica-sample/x.csv", True),
    ("solution/ok.py", False),
    ("anywhere/" + MATCH + "tactical_atd.txt", True),
    (MATCH + "video.mp4", True),
    ("evaluation/gt/" + MATCH + "possession_gt_public.csv", False),       # public, upstream's
    ("evaluation/gt/" + MATCH + "possession_gt_public_meta.json", False),
    ("evaluation/gt/" + MATCH + "possession_gt.csv", True),               # the private name
])
def test_hook_refuses_match_data_and_allows_the_rest(repo, path, refused):
    stage(repo, path)
    done = git(repo, "commit", "-q", "-m", "attempt")
    assert (done.returncode != 0) == refused, done.stderr
    if refused:
        assert "COMMIT REFUSED" in done.stderr and path in done.stderr
        assert git(repo, "rev-parse", "--verify", "-q", "HEAD").returncode != 0  # no commit


def test_hook_lists_every_refused_file_with_its_reason(repo):
    for path in ("data/a.txt", "out/b.txt", "solution/fine.py", "x/" + MATCH + "c.txt"):
        stage(repo, path)
    done = git(repo, "commit", "-q", "-m", "attempt")
    assert done.returncode != 0
    assert done.stderr.count("refused because") == 3
    for fragment in ("data/a.txt", "inside data/", "out/b.txt", "inside out/",
                     MATCH + "c.txt", "name starts with"):
        assert fragment in done.stderr
    assert "solution/fine.py" not in done.stderr


def test_removing_a_data_file_from_the_repository_is_allowed(repo):
    stage(repo, "data/old.txt")
    assert git(repo, "commit", "-q", "--no-verify", "-m", "setup").returncode == 0
    assert git(repo, "rm", "-q", "--cached", "data/old.txt").returncode == 0
    done = git(repo, "commit", "-q", "-m", "remove it")
    assert done.returncode == 0, done.stderr


def test_hook_file_has_unix_line_endings():
    with open(HOOK, "rb") as handle:
        content = handle.read()
    assert content.startswith(b"#!/bin/sh\n") and b"\r" not in content
