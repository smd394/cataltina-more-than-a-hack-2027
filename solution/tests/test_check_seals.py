import os
import shutil
import subprocess

import pytest

import check_seals
from conftest import write_csv

N = 30
GIT = shutil.which("git")
EVERY_CLONE = ["data/*", "!data/README.md", "out/", "external/", ".venv-vision/"]
THIS_REPO = ["/NOTES.md", "/.tools/", "/docs/private/", "/drafts/"]     # invented extra paths
EXTRA = ["NOTES.md", ".tools/x", "docs/private/x", "drafts/x"]


def failing(rows):
    return sorted(seal for seal, ok, _ in rows if not ok)


def test_valid_file_passes_seals_1_to_4(tmp_path):
    path = write_csv(tmp_path / "good.csv", ["H"] * 10 + ["A"] * 10 + ["D"] * 5 + ["X"] * 5)
    assert failing(check_seals.file_seals(path, n_expected=N)) == []


@pytest.mark.parametrize("name, labels, frames, expected", [
    ("short", ["H"] * (N - 1), None, ["1 rows", "2 frame order", "4 validator"]),
    ("off by one", ["H"] * N, range(1, N + 1), ["2 frame order", "4 validator"]),
    ("bad label", ["H"] * (N - 1) + ["Z"], None, ["3 labels", "4 validator"]),
    ("blank label", ["H"] * (N - 1) + [""], None, ["3 labels", "4 validator"]),
])
def test_broken_file_fails_the_right_seals(tmp_path, name, labels, frames, expected):
    path = write_csv(tmp_path / "bad.csv", labels, frames)
    assert failing(check_seals.file_seals(path, n_expected=N)) == expected


def test_code_seals_catch_a_retyped_count_and_label_list(tmp_path):
    # Built at runtime so this test file itself never contains what the seal searches for.
    count = "199" + "319"
    vocabulary = ", ".join('"{}"'.format(label) for label in "HAD")
    (tmp_path / "planted.py").write_text("N = {}\nL = ({})\n".format(count, vocabulary))
    (tmp_path / "notes.md").write_text(count)          # prose is not code: ignored
    assert failing(check_seals.code_seals(str(tmp_path))) == ["1 rows (code)", "3 labels (code)"]


def test_code_seals_pass_on_clean_code(tmp_path):
    (tmp_path / "clean.py").write_text("from common import LABELS, N_FRAMES\n")
    assert failing(check_seals.code_seals(str(tmp_path))) == []


def throwaway_repo(tmp_path, ignore_lines):
    """A fresh git repository with this .gitignore and our pre-commit hook installed."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([GIT, "init", "-q"], cwd=str(repo), check=True)
    (repo / ".gitignore").write_text("\n".join(ignore_lines) + "\n")
    shutil.copyfile(os.path.join(check_seals.HERE, "hooks", "pre-commit"),
                    str(repo / ".git" / "hooks" / "pre-commit"))
    return str(repo)


def extra_file(tmp_path, paths):
    path = tmp_path / "extra.txt"
    path.write_text("# a comment line\n\n" + "\n".join(paths) + "\n")
    return str(path)


@pytest.mark.skipif(GIT is None, reason="git is not installed")
@pytest.mark.parametrize("with_extra, expected", [(True, "8/8 ignored"), (False, "4/4 ignored")])
def test_privacy_seal_counts_the_extra_paths_only_when_their_file_exists(tmp_path, with_extra,
                                                                           expected):
    repo = throwaway_repo(tmp_path, EVERY_CLONE + THIS_REPO)
    extra = extra_file(tmp_path, EXTRA) if with_extra else str(tmp_path / "absent.txt")
    seal, ok, detail = check_seals.privacy_seal(root=repo, extra_file=extra)
    assert ok, detail
    assert detail.startswith(expected + ", control ok"), detail
    assert "pre-commit hook installed" in detail


@pytest.mark.skipif(GIT is None, reason="git is not installed")
@pytest.mark.parametrize("dropped, with_extra, expected", [
    ("external/", False, "3/4 ignored"),           # a path every clone must ignore
    ("/drafts/", True, "7/8 ignored"),             # a path the extra file asks for
])
def test_privacy_seal_fails_when_a_path_is_not_ignored(tmp_path, dropped, with_extra, expected):
    repo = throwaway_repo(tmp_path, [line for line in EVERY_CLONE + THIS_REPO if line != dropped])
    extra = extra_file(tmp_path, EXTRA) if with_extra else str(tmp_path / "absent.txt")
    seal, ok, detail = check_seals.privacy_seal(root=repo, extra_file=extra)
    assert not ok
    assert detail.startswith(expected), detail


def test_must_be_ignored_reads_one_path_a_line_and_skips_comments(tmp_path):
    assert check_seals.must_be_ignored(str(tmp_path / "absent.txt")) == \
        list(check_seals.MUST_BE_IGNORED)
    assert check_seals.must_be_ignored(extra_file(tmp_path, EXTRA)) == \
        list(check_seals.MUST_BE_IGNORED) + EXTRA
