"""Run only the tests relevant to the files changed since a git reference.

Usage (from the repository root):

    python scripts/select_tests.py                 # changes since HEAD, including uncommitted and untracked files
    python scripts/select_tests.py --base HEAD~1   # changes in the last commit and since
    python scripts/select_tests.py --list          # print the selection without running it
    python scripts/select_tests.py --tests-only    # only the changed and new test files
    python scripts/select_tests.py --all -n 4      # the full suite
    python scripts/select_tests.py -- -x -q        # pass further options to pytest

Changed source files select their categories (tests/categories.py) and the
categories that consume them; changed test files run themselves; a change to
``heuristics.py`` selects the categories of the modules that use the changed
constants; documentation and notebooks select nothing; a file no category
covers runs the full suite. The smoke tests always run with a selection.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.categories import (CATEGORIES, CONSUMERS, HEURISTICS, IGNORED_PREFIXES, IGNORED_SUFFIXES,  # noqa: E402
                              SELECTION_FILES, SMOKE, categories_of_source)


def changed_files(base: str) -> list[str]:
    """Return the repository paths changed since ``base``, with uncommitted and untracked files."""
    def git(*args: str) -> list[str]:
        """Return the lines of a git command's output."""
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return [line for line in out.splitlines() if line]

    files = set(git("diff", "--name-only", base)) | set(git("ls-files", "--others", "--exclude-standard"))
    return sorted(files)


def changed_constants(base: str) -> set[str]:
    """Return the names of constants added, removed or changed in heuristics.py since ``base``."""
    diff = subprocess.run(["git", "diff", "-U0", base, "--", HEURISTICS], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout
    names = set()
    for line in diff.splitlines():
        if line[:1] in "+-" and not line.startswith(("+++", "---")):
            m = re.match(r"[+-]([A-Z][A-Z0-9_]*)\s*=", line)
            if m:
                names.add(m.group(1))
    return names


def users_of(names: set[str]) -> set[str]:
    """Return the package and script paths that refer to any of the constant names."""
    if not names:
        return set()
    pattern = re.compile(r"\b(" + "|".join(sorted(names)) + r")\b")
    users = set()
    for path in list((ROOT / "worldgen").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel != HEURISTICS and pattern.search(path.read_text(encoding="utf-8")):
            users.add(rel)
    return users


def with_consumers(categories: set[str]) -> set[str]:
    """Return the categories with every category that consumes them, followed transitively."""
    out, todo = set(categories), list(categories)
    while todo:
        for consumer in CONSUMERS.get(todo.pop(), ()):
            if consumer not in out:
                out.add(consumer)
                todo.append(consumer)
    return out


def select(files: list[str], base: str) -> tuple[set[str], set[str], list[str]]:
    """Return the selected categories, extra test modules, and paths no category covers."""
    categories: set[str] = set()
    tests: set[str] = set()
    uncovered: list[str] = []
    for path in files:
        if path.startswith(IGNORED_PREFIXES) or path.endswith(IGNORED_SUFFIXES):
            continue
        if path in SELECTION_FILES:
            tests.add("test_categories")
            continue
        if path.startswith("tests/"):
            name = Path(path).stem
            if name.startswith("test_") and (ROOT / path).exists():
                tests.add(name)
            elif name in ("conftest", "__init__"):
                uncovered.append(path)
            continue
        if path == HEURISTICS:
            users = users_of(changed_constants(base))
            for user in users:
                categories.update(categories_of_source(user))
            continue
        found = categories_of_source(path)
        if found:
            categories.update(found)
        else:
            uncovered.append(path)
    return with_consumers(categories), tests, uncovered


def test_modules(categories: set[str], tests: set[str]) -> list[str]:
    """Return the test files to run for the categories and extra modules, smoke tests included."""
    names = set(tests)
    for name in categories:
        names.update(CATEGORIES[name]["tests"])
    if names:
        names.update(SMOKE)
    return [f"tests/{n}.py" for n in sorted(names) if (ROOT / "tests" / f"{n}.py").exists()]


def main() -> int:
    """Select and run the tests for the current changes; return pytest's exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="HEAD", help="git reference to compare with (default HEAD)")
    parser.add_argument("--all", action="store_true", help="run the full suite")
    parser.add_argument("--list", action="store_true", help="print the selection and stop")
    parser.add_argument("--tests-only", action="store_true", help="run only the changed and new test files")
    parser.add_argument("-n", default=None, help="pytest-xdist workers (e.g. 4 or auto)")
    parser.add_argument("pytest_args", nargs="*", help="further pytest options (after --)")
    args = parser.parse_args()

    if args.all:
        targets, reason = ["tests"], "full suite (requested)"
    else:
        files = changed_files(args.base)
        categories, tests, uncovered = select(files, args.base)
        if args.tests_only:
            targets = [f"tests/{n}.py" for n in sorted(tests)]
            reason = f"changed tests only: {', '.join(sorted(tests)) or 'none'}"
        elif uncovered:
            targets, reason = ["tests"], "full suite: no category covers " + ", ".join(uncovered)
        else:
            targets = test_modules(categories, tests)
            reason = (f"categories: {', '.join(sorted(categories)) or 'none'}; "
                      f"changed tests: {', '.join(sorted(tests)) or 'none'}")
        print(f"{len(files)} changed files since {args.base}")
    print(reason)
    if not targets:
        print("nothing to test")
        return 0
    command = [sys.executable, "-m", "pytest", *targets, *(["-n", args.n] if args.n else []), *args.pytest_args]
    print(" ".join(command[2:]))
    if args.list:
        return 0
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
