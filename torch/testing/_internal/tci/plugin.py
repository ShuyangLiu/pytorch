"""The pytest side of tci: test/conftest.py's --list-platforms and --list-platforms-json
collect the tests and print where and when each one runs, instead of running them."""

from __future__ import annotations

import functools
import json
import sys
from collections import Counter, defaultdict
from typing import Any, TextIO, TYPE_CHECKING

from _pytest.main import wrap_session

from torch.testing._internal import tci
from torch.testing._internal.common_utils import TestCase


if TYPE_CHECKING:
    import pytest
    from _pytest.config import Config


def list_platforms(config: Config) -> int:
    """A pytest session that lists the tests it collects and runs none of them."""
    # The listing has stdout to itself: the terminal reporter, which the session's
    # pytest_configure makes, writes pytest's own lines to stderr.
    stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        return wrap_session(config, functools.partial(_list, stdout=stdout))
    finally:
        sys.stdout = stdout


def _list(config: Config, session: pytest.Session, stdout: TextIO) -> None:
    """Prints where and when each collected test runs, once for each group of tests
    that run alike, as text or JSON."""
    session.perform_collect()
    as_json = config.option.list_platforms_json
    # tci.describe()'s box drawing, or ASCII if stdout can't encode it (a code page)
    try:
        "┌─│".encode(stdout.encoding or "ascii")
        unicode = True
    except (LookupError, UnicodeEncodeError):
        unicode = False
    # Tests of one class share a spec unless they have statements of their own.
    described: dict[tuple[type, tuple[Any, ...]], str] = {}
    groups: dict[str, list[Any]] = defaultdict(list)
    no_spec = []
    for item in session.items:
        cls = getattr(item, "cls", None)
        if cls is None or not issubclass(cls, TestCase):
            no_spec.append(item)
            continue
        test = getattr(cls, item.name, None)
        key = (cls, vars(test).get(tci.ATTR, ()) if test is not None else ())
        if key not in described:
            spec = tci.resolve(cls, test)
            if as_json:
                described[key] = json.dumps(tci.explain(spec))
            else:
                # printed indented by 2, within 80 columns
                lines = tci.describe(spec, width=78, unicode=unicode)
                described[key] = "\n".join(lines)
        groups[described[key]].append(item)
    if as_json:
        listing = []
        for explained, items in groups.items():
            tests = [item.nodeid for item in items]
            listing.append({"tests": tests, **json.loads(explained)})
        if no_spec:  # tci says nothing about these
            tests = [item.nodeid for item in no_spec]
            listing.append({"tests": tests, "runs": None, "excluded": {}})
        print(json.dumps(listing, indent=2), file=stdout)
    else:
        _print_text(groups, no_spec, stdout)
    # Collection errors fail the run, as they do without the listing.
    if errors := session.testsfailed:
        plural = "s" if errors > 1 else ""
        raise session.Interrupted(f"{errors} error{plural} during collection")


def _print_text(groups: dict[str, list], no_spec: list, out: TextIO) -> None:
    # A class is named once, with the group that has most of its tests if that is more
    # than one; its other tests are named one by one.
    classes: dict[str, Counter[str]] = defaultdict(Counter)
    for description, items in groups.items():
        for item in items:
            classes[item.parent.nodeid][description] += 1
    legend = [
        "Each table says when tests run, then where: a row for each kind of platform.",
        "A platform matches a row if it meets every line of every cell in the row.",
        "Commas separate alternatives; any means any value.",
    ]
    print(*legend, sep="\n", file=out)
    for description, items in groups.items():
        names, summarized = [], set()
        for item in items:
            cls, counts = item.parent.nodeid, classes[item.parent.nodeid]
            count, total = counts[description], counts.total()
            if count == 1 or counts.most_common(1)[0][0] != description:
                names.append(item.nodeid)
            elif cls not in summarized:
                summarized.add(cls)
                some = f"{count:,} of {total:,}" if count < total else f"all {total:,}"
                names.append(f"{cls}: {some} tests")
        lines = [f"  {line}" for line in description.splitlines()]
        print("", *names, *lines, sep="\n", file=out)
    if no_spec:
        files = Counter(item.nodeid.split("::")[0] for item in no_spec)
        print("", "No spec, outside common_utils.TestCase:", sep="\n", file=out)
        print(*(f"  {f}: {n:,} tests" for f, n in files.items()), sep="\n", file=out)
