"""How tci works: selectors over platforms, the decorators' statements, folding them
into a Spec (resolve) and spelling a spec out (explain, describe). __init__.py has the
public API, and DEFAULT."""

from __future__ import annotations

import dataclasses
import functools
import itertools
import operator
import re
import types
import unittest
from typing import Any, Literal, TYPE_CHECKING, TypeVar

from torch.testing._internal.tci.vocabulary import DEVICES, ENUMS, HARNESSES, SCHEDULES


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from torch.testing._internal.torchci.environment import Environment

T = TypeVar("T")
Verb = Literal["when", "runs_on", "include", "exclude", "on"]


# ----------------------------------------------------------------------- selectors
class Selector:
    """A set of platforms. Combine with & | ~ -; a selector has no truth value."""

    def matches(self, env: Environment) -> bool:
        raise NotImplementedError

    def when(self, *cadences: Cadence) -> Timed:
        """These platforms at their own cadences, for include(), exclude() and on()."""
        return Timed(self, _cadences(cadences, "when"))

    def exclude(self, *others: Selector | Sequence[Selector]) -> Selector:
        """self - other for each of others: rocm.exclude(device_name.mi350x)."""
        out = self
        for other in _flatten(others):
            if not isinstance(other, Selector):
                raise TypeError(f".exclude() takes platforms, got {_what(other)}")
            out = out - other
        return out

    def __and__(self, other: object) -> Selector:
        if not isinstance(other, Selector):
            return NotImplemented
        return _satisfiable(And(self, other))

    def __or__(self, other: object) -> Selector:
        return Or(self, other) if isinstance(other, Selector) else NotImplemented

    def __sub__(self, other: object) -> Selector:
        return self & Not(other) if isinstance(other, Selector) else NotImplemented

    def __invert__(self) -> Selector:
        return Not(self)

    # Only reached when the left operand is not a selector: tci.device_count >= 2 & ...
    def __rand__(self, other: object) -> Selector:
        raise _operand_error("&", other)

    def __ror__(self, other: object) -> Selector:
        raise _operand_error("|", other)

    def __bool__(self) -> bool:
        raise TypeError(
            "a selector has no truth value: use & | ~, not and, or, not or if"
        )

    def __str__(self) -> str:
        return _show(self, 0)


@dataclasses.dataclass(frozen=True)
class AnyRun(Selector):
    def matches(self, env):
        return True


@dataclasses.dataclass(frozen=True)
class Is(Selector):
    """An enum column, a count or the harness has one of these values."""

    field: str
    values: tuple[str | int, ...]
    one_of: bool = False

    def matches(self, env):
        return _get(env, self.field) in self.values


@dataclasses.dataclass(frozen=True)
class Device(Selector):
    families: tuple[str, ...]
    one_of: bool = False

    def matches(self, env):
        return _family(env.device_name) in self.families


@dataclasses.dataclass(frozen=True)
class Compare(Selector):
    """A version (compared per component) or a count."""

    field: str
    op: str
    value: str | int

    def matches(self, env):
        actual = _get(env, self.field)
        if isinstance(self.value, int):
            return _OPS[self.op](actual, self.value)
        a, b = _version(actual), _version(self.value)
        if self.op in ("==", "!="):
            equal = a == b if a is not None and b is not None else actual == self.value
            return equal == (self.op == "==")
        return a is not None and b is not None and _OPS[self.op](a, b)


@dataclasses.dataclass(frozen=True)
class VersionOneOf(Selector):
    field: str
    values: tuple[str, ...]

    def matches(self, env):
        # "3.14" is 3.14 but not 3.14t: equal versions must agree on free-threading too
        actual = _get(env, self.field)
        version, free = _version(actual), actual.endswith("t")

        def same(v: str) -> bool:
            equal = version is not None and version == _version(v)
            return actual == v or (equal and free == v.endswith("t"))

        return any(same(v) for v in self.values)


@dataclasses.dataclass(frozen=True)
class FreeThreaded(Selector):
    def matches(self, env):
        return env.python_version.endswith("t")


@dataclasses.dataclass(frozen=True, eq=False)
class EnvVar(Selector):
    """tci.env_vars.NAME, for any NAME: the variable is "1"; compare it for others."""

    name: str

    def matches(self, env):
        return env.flags.get(self.name) == "1"

    def _value(self, value: object) -> str:
        if not isinstance(value, str):
            name = f"tci.env_vars.{self.name}"
            raise TypeError(f'{name} compares to a string, e.g. ({name} == "1")')
        return value

    # Another selector compares by identity, as in _conj's checks against ANY.
    def __eq__(self, value):  # type: ignore[override]
        if isinstance(value, Selector):
            return NotImplemented
        return EnvIs(self.name, (self._value(value),))

    def __ne__(self, value):  # type: ignore[override]
        if isinstance(value, Selector):
            return NotImplemented
        return Not(EnvIs(self.name, (self._value(value),)))

    def __hash__(self) -> int:
        return hash(("env_vars", self.name))

    def one_of(self, *values: str) -> Selector:
        return EnvIs(self.name, tuple(self._value(v) for v in values), one_of=True)


@dataclasses.dataclass(frozen=True)
class EnvIs(Selector):
    name: str
    values: tuple[str, ...]
    one_of: bool = False

    def matches(self, env):
        return env.flags.get(self.name, "") in self.values


@dataclasses.dataclass(frozen=True)
class And(Selector):
    a: Selector
    b: Selector

    def matches(self, env):
        return self.a.matches(env) and self.b.matches(env)


@dataclasses.dataclass(frozen=True)
class Or(Selector):
    a: Selector
    b: Selector

    def matches(self, env):
        return self.a.matches(env) or self.b.matches(env)


@dataclasses.dataclass(frozen=True)
class Not(Selector):
    a: Selector

    def matches(self, env):
        return not self.a.matches(env)


@dataclasses.dataclass(frozen=True)
class Named(Selector):
    name: str
    body: Selector

    def matches(self, env):
        return self.body.matches(env)


ANY: Selector = AnyRun()
NOTHING: Selector = Not(ANY)


def named(name: str, body: Selector) -> Selector:
    """A selector that prints as its name, as specs and lint messages show it."""
    return Named(name, body)


def _satisfiable(sel: Selector) -> Selector:
    """A conjunction that needs two values of one setting at once can never match."""
    by_field: dict[str, list[Is | Device]] = {}

    def walk(x: Selector) -> None:
        if isinstance(x, Named):
            walk(x.body)
        elif isinstance(x, And):
            walk(x.a)
            walk(x.b)
        elif isinstance(x, Is):
            by_field.setdefault(x.field, []).append(x)
        elif isinstance(x, Device):
            by_field.setdefault("device_name", []).append(x)

    walk(sel)
    for field, atoms in by_field.items():
        sets = [set(a.values if isinstance(a, Is) else a.families) for a in atoms]
        if not set.intersection(*sets):
            names = [f"tci.{a}" for a in atoms]
            both, either = " & ".join(names), " | ".join(names)
            which = "either" if len(names) == 2 else "any of them"
            raise ValueError(
                f"{both} can never match: a run has exactly one {field}; "
                f"for {which}, write {either}"
            )
    return sel


# ------------------------------------------------------------------------ settings
class Setting:
    def __init__(self, name: str) -> None:
        self.name = name


class EnumSetting(Setting):
    def __init__(self, name: str, values: tuple[str, ...]) -> None:
        super().__init__(name)
        self.values = values

    def __getattr__(self, value: str) -> Selector:
        if value not in self.values:
            expected = ", ".join(self.values)
            raise AttributeError(
                f"tci.{self.name} has no value '{value}'; expected one of {expected}"
            )
        return Is(self.name, (value,))

    def one_of(self, *values: str) -> Selector:
        for value in values:
            getattr(self, value)
        return Is(self.name, values, one_of=True)


class DeviceSetting(Setting):
    def __getattr__(self, family: str) -> Selector:
        if family not in DEVICES:
            expected = ", ".join(DEVICES)
            raise AttributeError(
                f"tci.device_name has no family '{family}'; expected one of {expected}"
            )
        return Device((family,))

    def one_of(self, *families: str) -> Selector:
        for family in families:
            getattr(self, family)
        return Device(families, one_of=True)


class OrderedSetting(Setting):
    """A version (a string, compared per component) or a count (an int)."""

    def __init__(self, name: str, kind: type) -> None:
        super().__init__(name)
        self.kind = kind

    def _compare(self, op: str, value: object) -> Selector:
        if not isinstance(value, self.kind) or isinstance(value, bool):
            if self.kind is str:
                what, example = "a version string", f'(tci.{self.name} >= "3.12")'
            else:
                what, example = "a count", f"(tci.{self.name} >= 2)"
            hint = "; as a float, 3.10 would be 3.1" if isinstance(value, float) else ""
            raise TypeError(f"tci.{self.name} compares to {what}, e.g. {example}{hint}")
        return Compare(self.name, op, value)

    def __eq__(self, value):  # type: ignore[override]
        return self._compare("==", value)

    def __ne__(self, value):  # type: ignore[override]
        return self._compare("!=", value)

    def __lt__(self, value):
        return self._compare("<", value)

    def __le__(self, value):
        return self._compare("<=", value)

    def __gt__(self, value):
        return self._compare(">", value)

    def __ge__(self, value):
        return self._compare(">=", value)

    __hash__ = None  # type: ignore[assignment]

    def one_of(self, *values: str | int) -> Selector:
        for v in values:
            self._compare("==", v)
        if self.kind is int:
            return Is(self.name, values, one_of=True)
        return VersionOneOf(self.name, values)  # type: ignore[arg-type]

    @property
    def free_threaded(self) -> Selector:
        if self.name != "python_version":
            raise AttributeError(f"tci.{self.name} has no attribute 'free_threaded'")
        return FreeThreaded()


class EnvVars:
    """tci.env_vars: any variable, by its real name. Only harnesses set them in CI."""

    def __getattr__(self, name: str) -> EnvVar:
        if name.startswith("__"):  # copy, pickle and inspect probe these
            raise AttributeError(name)
        return EnvVar(name)


# One setting per identity column of torchci's Environment (test_tci.py keeps them in
# sync), plus the harness of the run. `os` shadows the stdlib module on purpose: tci.os.
os = EnumSetting("os", ENUMS["os"])
os_version = OrderedSetting("os_version", str)
cpu_architecture = EnumSetting("cpu_architecture", ENUMS["cpu_architecture"])
cpu_capability = EnumSetting("cpu_capability", ENUMS["cpu_capability"])
python_version = OrderedSetting("python_version", str)
cc_compiler = EnumSetting("cc_compiler", ENUMS["cc_compiler"])
cc_compiler_version = OrderedSetting("cc_compiler_version", str)
accelerator = EnumSetting("accelerator", ENUMS["accelerator"])
accelerator_version = OrderedSetting("accelerator_version", str)
device_count = OrderedSetting("device_count", int)
device_name = DeviceSetting("device_name")
device_capability = OrderedSetting("device_capability", str)
harness = EnumSetting("harness", tuple(HARNESSES))
env_vars = EnvVars()


# ------------------------------------------------------------------------ cadences
@dataclasses.dataclass(frozen=True)
class Cadence:
    """A value of SCHEDULES, listed (tci.when(tci.pull, tci.trunk)), never combined."""

    name: str

    def _combine(self, other: object) -> Cadence:
        raise TypeError(
            "cadences do not combine with & or |: list them, tci.when(tci.pull, "
            "tci.trunk), and give platforms their own with X.when(tci.periodic)"
        )

    __and__ = __rand__ = __or__ = __ror__ = _combine

    def __bool__(self) -> bool:
        raise TypeError("a cadence has no truth value")

    def __repr__(self) -> str:
        return f"tci.{self.name}"


pull, trunk, periodic = (Cadence(s) for s in SCHEDULES)


@dataclasses.dataclass(frozen=True)
class Timed:
    """Platforms with their own cadences, X.when(tci.periodic): include() and exclude()
    add or remove exactly those (platform, cadence) pairs, and on() re-times the
    platforms."""

    platforms: Selector
    cadences: frozenset[str]

    def when(self, *cadences: Cadence) -> Timed:
        raise TypeError(
            f"{self.platforms} already has its cadences; list them all in one .when()"
        )

    def exclude(self, *others: object) -> Timed:
        raise TypeError(
            "leave platforms out before giving them a cadence: X.exclude(Y).when(...)"
        )

    def _combine(self, *other: object) -> Timed:
        raise TypeError(
            "combine platforms before giving them a cadence, "
            "e.g. (A & B).when(tci.periodic)"
        )

    __and__ = __rand__ = __or__ = __ror__ = __sub__ = __rsub__ = __invert__ = _combine

    def __bool__(self) -> bool:
        raise TypeError("platforms with a cadence have no truth value")

    def __str__(self) -> str:
        return _show_timed(self.platforms, self.cadences)


# ---------------------------------------------------------------------- decorators
@dataclasses.dataclass(frozen=True)
class Statement:
    verb: Verb
    items: tuple[Selector | Timed, ...] = ()
    cadences: frozenset[str] = frozenset()  # when's
    reason: str = ""


# Statements live in the decorated object's own __dict__: a subclass does not inherit
# its bases' statements as its own, and resolve() visits each class of the MRO once.
# functools.wraps copies __dict__, so device-type and parametrized variants keep their
# function's.
ATTR = "__tci__"


@dataclasses.dataclass(frozen=True)
class Statements:
    """A decorator that attaches its statements, in order. It chains, one statement
    after another: tci.runs_on(tci.MULTI_GPU).exclude(tci.accelerator.rocm, reason=...).
    """

    statements: tuple[Statement, ...] = ()

    def _then(self, statement: Statement) -> Statements:
        if self.statements and self.statements[0].verb == "when":
            raise TypeError(
                "@tci.when(...) gives the whole test its cadences and stands alone: "
                "stack it above the platforms' decorators; platforms take their own "
                "cadence as X.when(tci.periodic)"
            )
        return Statements((*self.statements, statement))

    def when(self, *cadences: Cadence) -> Statements:
        if [st.verb for st in self.statements] == ["when"]:
            raise TypeError(
                "the test's cadences go in one when(): tci.when(tci.pull, tci.trunk)"
            )
        raise TypeError(
            "a chain takes no cadence: give platforms their own, X.when(tci.periodic), "
            "or the test its cadences in a decorator of their own, @tci.when(...)"
        )

    def runs_on(self, *platforms: Selector | Sequence[Selector]) -> Statements:
        items = _platforms(platforms, "runs_on", timed=False)
        return self._then(Statement("runs_on", items))

    def include(
        self, *platforms: Selector | Timed | Sequence[Selector | Timed]
    ) -> Statements:
        return self._then(Statement("include", _platforms(platforms, "include")))

    def exclude(
        self, *platforms: Selector | Timed | Sequence[Selector | Timed], reason: str
    ) -> Statements:
        items = _platforms(platforms, "exclude")
        return self._then(Statement("exclude", items, reason=reason))

    def on(self, *platforms: Timed | Sequence[Timed]) -> Statements:
        items = _platforms(platforms, "on")
        for x in items:
            if not isinstance(x, Timed):
                example = f"on({_show(x, 4)}.when(tci.periodic))"
                raise TypeError(f"on() gives platforms their own cadences: {example}")
        return self._then(Statement("on", items))

    def __call__(self, obj: T) -> T:
        setattr(obj, ATTR, (*self.statements, *vars(obj).get(ATTR, ())))
        return obj


def when(*cadences: Cadence) -> Statements:
    """The test's cadences, for all of its platforms; none runs more often."""
    return Statements((Statement("when", cadences=_cadences(cadences, "when")),))


def runs_on(*platforms: Selector | Sequence[Selector]) -> Statements:
    """Replace the inherited platforms; pairs added on the others go too."""
    return Statements().runs_on(*platforms)


def include(*platforms: Selector | Timed | Sequence[Selector | Timed]) -> Statements:
    """Add platforms at the test's cadences; X.when(...) adds just those pairs."""
    return Statements().include(*platforms)


def exclude(
    *platforms: Selector | Timed | Sequence[Selector | Timed], reason: str
) -> Statements:
    """Leave platforms out; X.when(...) removes just those pairs. The HUD shows why."""
    return Statements().exclude(*platforms, reason=reason)


def on(*platforms: Timed | Sequence[Timed]) -> Statements:
    """Re-time platforms the test runs, at this level and below; adds none."""
    return Statements().on(*platforms)


# ---------------------------------------------------------------------------- fold
@dataclasses.dataclass(frozen=True)
class Spec:
    """Where and when a test runs: its platforms at its cadences, platforms on()
    re-timed, and (platform, cadence) pairs added and removed, per cadence. Every part
    is a selector over platforms, so a spec resolves without knowing which exist."""

    platforms: Selector
    # every platform's unless on() says otherwise, and the ceiling
    cadences: frozenset[str]
    # the last one that names a platform decides it
    retimed: tuple[tuple[Selector, frozenset[str]], ...]
    added: Mapping[str, Selector]
    removed: Mapping[str, Selector]
    # each exclude's platforms and reason, root first, to tell people why
    excludes: tuple[tuple[Selector | Timed, str], ...]

    def cadences_of(self, env: Environment) -> frozenset[str]:
        top = _top(self.cadences)
        out: frozenset[str] = frozenset()
        if self.platforms.matches(env):
            retimed = (c for sel, c in reversed(self.retimed) if sel.matches(env))
            out = next(retimed, self.cadences)
            # a test that also runs periodically does so on all of its platforms
            if "periodic" in self.cadences and top < SCHEDULES.index("periodic"):
                out |= {"periodic"}
        out |= {s for s, sel in self.added.items() if sel.matches(env)}
        removed = {s for s, sel in self.removed.items() if sel.matches(env)}
        return _cap(out, top) - removed

    def __str__(self) -> str:
        cadences = ", ".join(s for s in SCHEDULES if s in self.cadences)
        parts = [str(self.platforms), f"when({cadences})"]
        parts += [f"on({_show_timed(sel, c)})" for sel, c in self.retimed]
        parts += _show_pairs("+", self.added) + _show_pairs("-", self.removed)
        return "; ".join(parts)


def resolve(cls: type, test: Callable[..., object] | None = None) -> Spec:
    """Fold a test's statements: the root, cls's MRO base first, then the test.

    The root is the base-most TestCase with a runs_on (common_utils.TestCase); a
    decorated mixin folds in after it, at its place in the MRO, whatever the order of
    the bases.
    """
    levels = [c for c in reversed(cls.__mro__) if vars(c).get(ATTR)]
    roots = [c for c in levels if any(st.verb == "runs_on" for st in vars(c)[ATTR])]
    root = next((c for c in roots if issubclass(c, unittest.TestCase)), None)
    if root is None:
        raise TypeError(
            f"{cls.__qualname__}: no TestCase in its MRO has a @tci.runs_on"
        )
    fold = _Fold()
    for level in [root, *(c for c in levels if c is not root)]:
        fold.level(vars(level)[ATTR], level.__qualname__)
    if test is not None:
        fold.level(vars(test).get(ATTR, ()), test.__qualname__)
    retimed, excludes = tuple(fold.retimed), tuple(fold.excludes)
    added, removed = dict(fold.added), dict(fold.removed)
    return Spec(fold.platforms, fold.cadences, retimed, added, removed, excludes)


class _Fold:
    """Statements in order, top to bottom and left to right, each level after the one
    above: where two overlap, the later one wins."""

    def __init__(self) -> None:
        self.platforms = NOTHING
        self.cadences = frozenset(SCHEDULES)
        self.retimed: list[tuple[Selector, frozenset[str]]] = []
        self.added = dict.fromkeys(SCHEDULES, NOTHING)
        self.removed = dict.fromkeys(SCHEDULES, NOTHING)
        self.excludes: list[tuple[Selector | Timed, str]] = []

    def level(self, statements: Iterable[Statement], name: str) -> None:
        verbs = [st.verb for st in statements]
        if verbs.count("runs_on") > 1 or verbs.count("when") > 1:
            raise TypeError(
                f"{name}: at most one @tci.runs_on and one @tci.when per level"
            )
        for st in statements:
            if st.verb == "when":
                self.cadences = st.cadences
            elif st.verb == "runs_on":
                # pairs added on the platforms it leaves out go with them
                x = functools.reduce(_disj, st.items)  # type: ignore[arg-type]
                self.platforms = x
                self.added = {s: _conj(a, x) for s, a in self.added.items()}
            else:
                if st.verb == "exclude":
                    self.excludes += [(item, st.reason) for item in st.items]
                for item in st.items:
                    if st.verb == "on":
                        self.on(item)  # type: ignore[arg-type]
                    elif isinstance(item, Timed):
                        self.pairs(item, add=st.verb == "include")
                    elif st.verb == "include":
                        self.platforms = _disj(self.platforms, item)
                    else:
                        out = Not(item)
                        self.platforms = _conj(self.platforms, out)
                        self.added = {s: _conj(a, out) for s, a in self.added.items()}

    def pairs(self, item: Timed, add: bool) -> None:
        grow, shrink = (self.added, self.removed) if add else (self.removed, self.added)
        for s in item.cadences:
            grow[s] = _disj(grow[s], item.platforms)
            shrink[s] = _conj(shrink[s], Not(item.platforms))

    def on(self, item: Timed) -> None:
        # Platforms the test has only through added pairs become its own, and re-timing
        # a platform the test runs lifts the pairs removed above at those cadences.
        x, paired = item.platforms, functools.reduce(_disj, self.added.values())
        own = _disj(self.platforms, paired)
        self.platforms = _disj(self.platforms, _conj(x, paired))
        self.retimed.append((x, item.cadences))
        self.added = {s: _conj(a, Not(x)) for s, a in self.added.items()}
        for s in item.cadences:
            self.removed[s] = _conj(self.removed[s], Not(_conj(x, own)))


# ------------------------------------------------------------------------ describe
def explain(spec: Spec) -> dict[str, Any]:
    """Where and when a spec runs, as JSON-ready data: {"runs": [{"when": [cadence,
    ...], "on": [platform, ...]}, ...], "excluded": {platforms: reason}}. Each platform
    is a kind of platform: the settings it constrains, named selectors spelled out; a
    setting it doesn't list can be any value. A setting is a list of the values it may
    have, or {"not": [values it may not have], "checks": [checks it passes]}, a check
    being a comparison or "free-threaded", either after "not" if it must fail, such as
    ">= 3.15" or "not free-threaded"."""
    runs = []
    for cadences, rows in _described(spec):
        platforms = [{key: _json(key, values) for key, values in row} for row in rows]
        runs.append({"when": cadences, "on": platforms})
    excluded = {str(platforms): reason for platforms, reason in spec.excludes}
    return {"runs": runs, "excluded": excluded}


def _json(key: str, values: _Values) -> list[Any] | dict[str, list[Any]]:
    """values as explain() gives them: the values allowed, or those left out and the
    checks the rest pass."""
    if values.allowed is not None:
        return _ordered(key, values.allowed)
    out: dict[str, list[Any]] = {}
    if values.excluded:
        out["not"] = _ordered(key, values.excluded)
    if values.tests:
        out["checks"] = sorted(_test(sel, holds) for sel, holds in values.tests)
    return out


def describe(spec: Spec, width: int = 80, unicode: bool = True) -> list[str]:
    """Where and when a spec runs, for people, in lines that fit width: for each cadence
    a table, When on top and a row for each kind of platform, then why each exclude
    leaves platforms out. unicode=False draws the tables in ASCII."""
    lines = []
    for cadences, rows in _described(spec):
        lines += _table(_say(cadences), [dict(row) for row in rows], width)
    if not lines:
        lines.append("Runs on no platform.")
    excluded = {str(platforms): reason for platforms, reason in spec.excludes}
    for platforms, reason in excluded.items():
        lines += _wrap(f"Excluded {platforms}: {reason}", width)
    return lines if unicode else [line.translate(_ASCII_BORDERS) for line in lines]


def _described(spec: Spec) -> list[tuple[list[str], list[_Key]]]:
    """Each cadence's kinds of platform, cadences with the same kinds together."""
    groups: dict[frozenset[_Key], tuple[list[str], list[_Key]]] = {}
    for s, region in _regions(spec).items():
        rows = [tuple(sorted(row.items(), key=_order)) for row in _rows(region)]
        if rows:
            groups.setdefault(frozenset(rows), ([], rows))[0].append(s)
    return list(groups.values())


# describe()'s columns and the settings each shows: how the tests run, then the
# machine, then the build. Env vars have a column of their own.
_TABLE = {
    "Harness": ("harness",),
    "OS": ("os", "os_version"),
    "CPU": ("cpu_architecture", "cpu_capability"),
    "Accelerator": (
        "accelerator",
        "accelerator_version",
        "device_name",
        "device_count",
    ),
    "Python": ("python_version",),
    "Compiler": ("cc_compiler", "cc_compiler_version"),
}
# The columns a table always shows, "any" where a platform may have any value: a
# platform is a whole machine. The others show only where some row narrows them.
_WHOLE = ("Harness", "OS", "CPU", "Accelerator", "Python")


def _table(when: str, rows: list[dict[str, _Values]], width: int) -> list[str]:
    """A cadence's table: When on top, then a row for each kind of platform, a column
    for each part of _WHOLE and any other part some row narrows, with "any" where a row
    doesn't, and in a cell a line for each condition. The rows of a harness are a group
    beside it. Env vars every row is within, as when its harness sets them so, go on top
    too."""
    shared: list[str] = []
    for key in sorted({k for row in rows for k in row if k.startswith("env_vars.")}):
        values = next(row[key] for row in rows if key in row)
        if all(_meet(row, {key: values}) == row for row in rows):
            shared += _lines(key, values)
            hoisted = (key, values)
            rows = [dict(kv for kv in row.items() if kv != hoisted) for row in rows]
    cells = [{c: _cell(row, keys) for c, keys in _TABLE.items()} for row in rows]
    for cell, row in zip(cells, rows):
        cell["Env"] = _cell(row, [key for key in row if key.startswith("env_vars.")])
    narrowed = [c for c in cells[0] if any(cell[c] != "any" for cell in cells)]
    columns = [c for c in cells[0] if narrowed and (c in _WHOLE or c in narrowed)]
    where = "" if columns else "any platform"
    top = {"When:": when, "Where:": where, "Env:": "\n".join(shared)}
    top = {label: text for label, text in top.items() if text}
    if not columns:
        heading = _heading(top, width - 4)
        inner = max(map(len, heading))
        box = [_rule("┌", "─", "┐", [inner]), *(_boxed(x, inner) for x in heading)]
        return [*box, _rule("└", "─", "┘", [inner])]
    harnesses = list(dict.fromkeys(cell["Harness"] for cell in cells))
    cells.sort(key=lambda cell: harnesses.index(cell["Harness"]))
    texts = [[cell[c] for c in columns] for cell in cells]
    # The rows of a harness are one group, the harness a cell beside them all.
    grouped = columns[0] == "Harness"
    groups: list[tuple[str, list[list[str]]]] = []
    for first, *rest in texts:
        if grouped and groups and groups[-1][0] == first:
            groups[-1][1].append(rest)
        else:
            groups.append((first, [rest]))

    def lines(row: list[str], widths: list[int]) -> list[list[str]]:
        wrapped = [_wrap(text, w) for text, w in zip(row, widths)]
        return [list(line) for line in itertools.zip_longest(*wrapped, fillvalue="")]

    def height(widths: list[int]) -> int:
        # a harness may run down beside its rows and the rules between them
        total = 0
        for first, rows in groups:
            beside = sum(len(lines(row, widths[1:])) + 1 for row in rows) - 1
            total += max(len(_wrap(first, widths[0])), beside)
        return total

    widths = _widths([columns, *texts], width, height)
    # as wide as what's on top, if that fits
    inner = sum(widths) + 3 * len(widths) - 3
    above = max(map(len, _heading(top, width - 4)))
    widths[-1] += max(above - inner, 0)
    inner = max(inner, above)

    def draw(cells: list[str]) -> str:
        line = "│ " + " │ ".join(map(str.ljust, cells, widths)) + " │"
        return line.replace(_NBSP, " ")

    out = [_rule("┌", "─", "┐", [inner])]
    out += [_boxed(line, inner) for line in _heading(top, inner)]
    out += [_rule("├", "┬", "┤", widths), *map(draw, lines(columns, widths))]
    for first, rows in groups:
        out.append(_rule("├", "┼", "┤", widths))
        beside: list[list[str] | None] = []
        for i, row in enumerate(rows):
            beside += [None] * (i > 0) + lines(row, widths[1:])
        harness = _wrap(first, widths[0])
        beside += [[""] * (len(widths) - 1)] * (len(harness) - len(beside))
        for line, text in itertools.zip_longest(beside, harness, fillvalue=""):
            if line is None:  # a rule between two of the harness's rows
                rule = _rule("├", "┼", "┤", widths[1:])
                out.append(f"│ {text.ljust(widths[0])} {rule}".replace(_NBSP, " "))
            else:
                out.append(draw([text, *line]))
    return out + [_rule("└", "┴", "┘", widths)]


def _heading(top: dict[str, str], width: int) -> list[str]:
    """What goes on top of a table, in lines of at most width: each label, then its
    text."""
    pad = max(map(len, top)) + 1
    lines = []
    for label, text in top.items():
        for i, line in enumerate(_wrap(text, width - pad)):
            lines.append((label if i == 0 else "").ljust(pad) + line)
    return lines


def _boxed(line: str, inner: int) -> str:
    return f"│ {line.ljust(inner)} │".replace(_NBSP, " ")


def _rule(start: str, cross: str, end: str, widths: list[int]) -> str:
    return start + cross.join("─" * (w + 2) for w in widths) + end


def _widths(rows: list[list[str]], width: int, height: Callable[..., int]) -> list[int]:
    """Widths for the columns of rows, the first the headers, so they fit width with
    their borders: as wide as their text or, if that's too wide, narrowed a column at a
    time, by what adds the fewest lines to height for each character it saves (the most
    characters, then the widest column, of those), down to what its text wraps to at
    spaces and slashes, then at each finer break of _BREAKS in turn; then no wider than
    their text wraps to."""
    columns = list(zip(*rows))
    widths = [max(len(x) for t in column for x in t.split("\n")) for column in columns]
    for level in range(1, len(_BREAKS)):
        floors = [max(_floor(text, level) for text in column) for column in columns]
        while (excess := sum(widths) + 3 * len(widths) + 1 - width) > 0:
            narrower = [
                [*widths[:i], w, *widths[i + 1 :]]
                for i, floor in enumerate(floors)
                for w in range(max(floor, widths[i] - excess), widths[i])
            ]
            if not narrower:
                break
            lines = height(widths)

            def cost(ws: list[int]) -> tuple[float, int, int]:
                saved = sum(widths) - sum(ws)
                return ((height(ws) - lines) / saved, -saved, max(ws))

            widths = min(narrower, key=cost)
    wrapped = [[x for t in col for x in _wrap(t, w)] for col, w in zip(columns, widths)]
    return [max(map(len, col)) for col in wrapped]


def _cell(row: dict[str, _Values], settings: Iterable[str]) -> str:
    """A cell for settings: a line for each condition row sets on them, or "any"."""
    lines = [line for key in settings if key in row for line in _lines(key, row[key])]
    return "\n".join(lines) or "any"


def _lines(key: str, values: _Values) -> list[str]:
    """The conditions values set on a setting, a line each: alternatives joined by
    commas, devices counted in words, an env var named. A line wraps after its commas
    and an env var's name: other spaces, as in "not linux" or "2+ devices", are
    no-break spaces."""
    texts = [_devices(values)] if key == "device_count" else _conditions(key, values)
    texts = [text.replace(" ", _NBSP).replace(f",{_NBSP}", ", ") for text in texts]
    if not key.startswith("env_vars."):
        return texts
    name = key.removeprefix("env_vars.")
    equals = values.allowed is not None
    return [f"{name}={text}" if equals else f"{name} {text}" for text in texts]


def _devices(values: _Values) -> str:
    """A count of devices in words, alternatives joined by commas."""
    if values.allowed == {0}:
        return "no device"
    if values.allowed is not None:
        counts = sorted(values.allowed)
        return f"{', '.join(map(str, counts))} device{'s' * (counts != [1])}"
    # counts from 0 up to the least one allowed read as a minimum
    least = next(n for n in itertools.count() if n not in values.excluded)
    if not least:
        return f"not {', '.join(map(str, sorted(values.excluded)))} devices"
    top = max(least, max(values.excluded) + 1)
    counts = [str(n) for n in range(least, top) if n not in values.excluded]
    return f"{', '.join([*counts, f'{top}+'])} devices"


def _wrap(text: str, width: int) -> list[str]:
    """text in lines of at most width: each of its lines broken at spaces, and a word
    too wide for a line at the finer breaks of _BREAKS it needs, the rest indented."""
    out = []
    for paragraph in text.split("\n"):
        lines: list[str] = []
        line = ""
        for word in re.findall(_BREAKS[0], paragraph):
            for i, piece in enumerate(_fitted(word, width - len(_INDENT))):
                joined = f"{line} {piece}" if line and i == 0 else line + piece
                room = width - len(_INDENT) if lines else width
                if line and len(joined.rstrip(_NBSP)) > room:
                    lines.append(line.rstrip(_NBSP))
                    line = piece
                else:
                    line = joined
        lines += [line.rstrip(_NBSP)] if line else []
        out += lines[:1] + [_INDENT + line for line in lines[1:]]
    return out


def _fitted(word: str, room: int, level: int = 1) -> list[str]:
    """word in pieces no wider than room, broken at the coarsest breaks that do it."""
    if len(word.rstrip(_NBSP)) <= room or level == len(_BREAKS):
        return [word]
    pieces = re.findall(_BREAKS[level], word)
    return [p for piece in pieces for p in _fitted(piece, room, level + 1)]


def _floor(text: str, level: int) -> int:
    """The narrowest text wraps to without breaking finer than _BREAKS[level]."""
    floors = [0]
    for line in text.split("\n"):
        pieces = [line]
        for pattern in _BREAKS[: level + 1]:
            pieces = [p for piece in pieces for p in re.findall(pattern, piece)]
        first, *rest = [len(p.rstrip(_NBSP)) for p in pieces] or [0]
        floors += [first, *(n + len(_INDENT) for n in rest)]
    return max(floors)


_NBSP = "\N{NO-BREAK SPACE}"
_INDENT = "  "
# Where a cell's text breaks, coarsest first, each only when the coarser ones can't fit
# it: at spaces, after slashes, at no-break spaces and after underscores, anywhere.
_BREAKS = ("[^ ]+", "[^/]*/|[^/]+", f"[^{_NBSP}_]*[{_NBSP}_]|[^{_NBSP}_]+", ".")


_ASCII_BORDERS = str.maketrans("┌┬┐├┼┤└┴┘─│", "+++++++++-|")


def _regions(spec: Spec) -> dict[str, Selector]:
    """The platforms that run at each cadence, as Spec.cadences_of decides it. Each
    part's cadences are capped first, so on()s that agree after the cap drop out."""
    top = _top(spec.cadences)
    soak = "periodic" in spec.cadences and top < SCHEDULES.index("periodic")
    out = dict.fromkeys(SCHEDULES, NOTHING)
    for i, s in enumerate(SCHEDULES[top:], top):
        if soak and s == "periodic":
            timed = ANY
        else:
            timed = ANY if s in _cap(spec.cadences, top) else NOTHING
            for sel, cadences in spec.retimed:
                if s in _cap(cadences, top):
                    timed = _disj(sel, timed)
                else:
                    timed = _conj(Not(sel), timed)
        # pairs added more often than the top cadence run at it
        added = [spec.added[x] for j, x in enumerate(SCHEDULES) if max(j, top) == i]
        region = functools.reduce(_disj, added, _conj(spec.platforms, timed))
        out[s] = _conj(region, Not(spec.removed[s]))
    return out


@dataclasses.dataclass(frozen=True)
class _Values:
    """The values of one setting that a row allows: those in allowed or, if it is None,
    any value but those in excluded that passes the tests, (selector, outcome) pairs
    that compare versions or check free-threading."""

    allowed: frozenset[Any] | None = None
    excluded: frozenset[Any] = frozenset()
    tests: frozenset[tuple[Selector, bool]] = frozenset()

    def holds(self, key: str, value: object) -> bool:
        run = _probe(**{key: value})
        return (
            (self.allowed is None or value in self.allowed)
            and value not in self.excluded
            and all(sel.matches(run) == out for sel, out in self.tests)
        )

    def meet(self, other: _Values, key: str) -> _Values | None:
        """The values both allow; None for none."""
        both = _Values(None, self.excluded | other.excluded, self.tests | other.tests)
        if self.allowed is not None and other.allowed is not None:
            candidates = self.allowed & other.allowed
        else:
            candidates = self.allowed if other.allowed is None else other.allowed
        if candidates is None:
            return both
        allowed = frozenset(v for v in candidates if both.holds(key, v))
        return _Values(allowed) if allowed else None

    def join(self, other: _Values) -> _Values | None:
        """The values either allows, if one _Values can say so."""
        if self.tests or other.tests:
            return None
        if self.allowed is not None and other.allowed is not None:
            return _Values(self.allowed | other.allowed)
        if self.allowed is None and other.allowed is None:
            return _Values(None, self.excluded & other.excluded)
        finite, cofinite = (self, other) if other.allowed is None else (other, self)
        return _Values(None, cofinite.excluded - finite.allowed)

    def negated(self) -> _Values:
        """The values an atom's _Values leaves out; an atom's has one of the parts."""
        if self.tests:
            [(sel, out)] = self.tests
            return _Values(tests=frozenset({(sel, not out)}))
        if self.allowed is not None:
            return _Values(None, self.allowed)
        return _Values(self.excluded)


# A kind of platform: the values of the settings it constrains.
_Row = dict[str, _Values]
_Key = tuple[tuple[str, _Values], ...]


def _rows(
    sel: Selector, positive: bool = True, within: _Row | None = None
) -> list[_Row]:
    """The platforms sel names, or if not positive leaves out, of those within names, as
    rows any of which matches: its disjunctive normal form, without rows that never
    match or that another row covers, and rows that differ in one setting merged."""
    within = within or {}
    if isinstance(sel, Named):
        return _rows(sel.body, positive, within)
    if isinstance(sel, Not):
        return _rows(sel.a, not positive, within)
    if isinstance(sel, AnyRun):
        return [within] if positive else []
    if isinstance(sel, (And, Or)):
        if isinstance(sel, And) == positive:
            # The side with fewer rows first, then the other within each of its rows:
            # ~DEFAULT alone has a product of its machines' rows, within one a handful.
            first, then = sorted((sel.a, sel.b), key=lambda x: _size(x, positive))
            rows = _rows(first, positive, within)
            return _simplify(r for row in rows for r in _rows(then, positive, row))
        return _simplify(r for x in (sel.a, sel.b) for r in _rows(x, positive, within))
    key, values = _atom(sel)
    values = values if positive else values.negated()
    if values.allowed == frozenset():
        return []
    row = _meet(within, {key: values}) if values != _Values() else within
    return [] if row is None else [row]


def _size(sel: Selector, positive: bool = True) -> int:
    """The most rows _rows can find for sel, before it drops or merges any."""
    if isinstance(sel, Named):
        return _size(sel.body, positive)
    if isinstance(sel, Not):
        return _size(sel.a, not positive)
    if isinstance(sel, (And, Or)):
        a, b = _size(sel.a, positive), _size(sel.b, positive)
        return a * b if isinstance(sel, And) == positive else a + b
    return 1


def _atom(sel: Selector) -> tuple[str, _Values]:
    """The setting an atom tests and the values it allows. A device's capability is its
    family's, so a capability names device families."""
    if isinstance(sel, Device):
        return "device_name", _Values(frozenset(sel.families))
    if isinstance(sel, FreeThreaded):
        return "python_version", _Values(tests=frozenset({(sel, True)}))
    if isinstance(sel, EnvVar):
        return f"env_vars.{sel.name}", _Values(frozenset({"1"}))
    if isinstance(sel, EnvIs):
        return f"env_vars.{sel.name}", _Values(frozenset(sel.values))
    if not isinstance(sel, (Is, Compare, VersionOneOf)):
        raise TypeError(f"not a selector atom: {sel!r}")
    if sel.field == "device_capability":
        named = (
            f
            for f, (name, _) in DEVICES.items()
            if sel.matches(_probe(device_name=name))
        )
        families = frozenset(named)
        if sel.matches(_probe(device_name="")):  # a device of no known family
            return "device_name", _Values(None, frozenset(DEVICES) - families)
        return "device_name", _Values(families)
    if isinstance(sel, Compare) and isinstance(sel.value, int):
        # every count above the value compares to it as value + 1 does
        op, value = _OPS[sel.op], sel.value
        below = range(max(value + 1, 0))
        passing = frozenset(n for n in below if op(n, value))
        if op(len(below), value):
            return sel.field, _Values(None, frozenset(below) - passing)
        return sel.field, _Values(passing)
    if isinstance(sel, Compare):
        return sel.field, _Values(tests=frozenset({(sel, True)}))
    return sel.field, _Values(frozenset(sel.values))


def _meet(x: _Row, y: _Row) -> _Row | None:
    """The row both match; None if none does."""
    out = dict(x)
    for key, values in y.items():
        both = out[key].meet(values, key) if key in out else values
        if both is None:
            return None
        out[key] = both
    return _narrowed(out)


def _narrowed(row: _Row) -> _Row | None:
    """row without the harnesses its env vars rule out, nor the env vars all its
    harnesses keep to; None if no harness is left. A harness sets its own variables
    and leaves the other harnesses' unset."""
    harness = row.get("harness")
    keys = [k for k in row if k.removeprefix("env_vars.") in _HARNESS_VARS]
    if harness is None or harness.allowed is None or not keys:
        return row

    def fits(h: str, key: str) -> list[bool]:
        """Whether row allows each value harness h may give the variable."""
        own = HARNESSES[h].get(key.removeprefix("env_vars."))
        return [row[key].holds(key, v) for v in ([own] if own else ["", "0"])]

    allowed = [h for h in harness.allowed if all(any(fits(h, k)) for k in keys)]
    if not allowed:
        return None
    implied = {k for k in keys if all(all(fits(h, k)) for h in allowed)}
    out = {k: v for k, v in row.items() if k not in implied}
    return out | {"harness": _Values(frozenset(allowed))}


_HARNESS_VARS = {name for variables in HARNESSES.values() for name in variables}


def _join(x: _Row, y: _Row) -> _Row | None:
    """The row either matches, if they differ in one setting."""
    differ = [key for key in x if x[key] != y.get(key)]
    if x.keys() != y.keys() or len(differ) != 1:
        return None
    key = differ[0]
    values = x[key].join(y[key])
    if values is None:
        return None
    out = {k: v for k, v in x.items() if k != key}
    return out | {key: values} if values != _Values() else out


def _simplify(rows: Iterable[_Row]) -> list[_Row]:
    rows = list(rows)
    rows = [
        r
        for i, r in enumerate(rows)
        if not any(_inside(r, i, o, j) for j, o in enumerate(rows) if j != i)
    ]
    while True:
        for i, j in itertools.combinations(range(len(rows)), 2):
            if (joined := _join(rows[i], rows[j])) is not None:
                break
        else:
            return rows
        rows[i] = joined
        del rows[j]
        # only the merged row is new: it is inside another, or others are inside it
        if any(_inside(joined, i, o, k) for k, o in enumerate(rows) if k != i):
            del rows[i]
        else:
            rows = [r for k, r in enumerate(rows) if not _inside(r, k, joined, i)]


def _inside(r: _Row, i: int, o: _Row, j: int) -> bool:
    """Whether row r, at i, adds nothing beside o, at j: it is inside o, or equal to it
    and after it. A row inside o has each setting o has but env vars, which harnesses
    may imply."""
    if any(k not in r for k in o if not k.startswith("env_vars.")):
        return False
    return _meet(r, o) == r and (j < i or _meet(o, r) != o)


# Settings in the order people read a platform, as describe()'s columns show it: how its
# tests run, the machine, then the build; env vars last.
_ORDER = (
    "harness",
    "os",
    "os_version",
    "cpu_architecture",
    "cpu_capability",
    "accelerator",
    "accelerator_version",
    "device_name",
    "device_count",
    "python_version",
    "cc_compiler",
    "cc_compiler_version",
)
_CADENCE_WORDS = {
    "pull": "every PR",
    "trunk": "every merge",
    "periodic": "periodically",
}
_VOCABULARY = {**ENUMS, "harness": tuple(HARNESSES), "device_name": tuple(DEVICES)}


def _order(item: str | tuple[str, object]) -> tuple[int, str]:
    key = item if isinstance(item, str) else item[0]
    return (_ORDER.index(key) if key in _ORDER else len(_ORDER), key)


def _conditions(key: str, values: _Values) -> list[str]:
    """The values allowed, or the tests they pass and those left out: alternatives
    joined by commas."""
    if values.allowed is not None:
        return [", ".join(_sorted(key, values.allowed))]
    parts = sorted(_test(sel, out) for sel, out in values.tests)
    left_out = set(values.excluded)
    if key == "device_count":
        # counts from 0 up to the least one allowed read as a minimum
        least = next(n for n in itertools.count() if n not in left_out)
        left_out -= set(range(least))
        if least:
            parts.append(f">= {least}")
    if left_out:
        parts.append("not " + ", ".join(_sorted(key, left_out)))
    return parts


def _test(sel: Selector, out: bool) -> str:
    text = f"{sel.op} {sel.value}" if isinstance(sel, Compare) else "free-threaded"
    return text if out else f"not {text}"


def _sorted(key: str, values: Iterable[Any]) -> list[str]:
    """Values as text, in order; "" is an unset env var."""
    return [str(v) or '""' for v in _ordered(key, values)]


def _ordered(key: str, values: Iterable[Any]) -> list[Any]:
    """Values in the vocabulary's order, or as versions."""
    vocab = _VOCABULARY.get(key)
    if vocab is not None:
        return sorted(values, key=vocab.index)
    return sorted(values, key=lambda v: (_version(str(v)) or (), str(v)))


def _say(cadences: Iterable[str]) -> str:
    words = [_CADENCE_WORDS[s] for s in SCHEDULES if s in cadences]
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def _top(cadences: Iterable[str]) -> int:
    """The most frequent cadence's index in SCHEDULES, or len(SCHEDULES) for none."""
    return min((SCHEDULES.index(s) for s in cadences), default=len(SCHEDULES))


def _cap(cadences: frozenset[str], top: int) -> frozenset[str]:
    """Nothing runs more often than SCHEDULES[top]: more frequent becomes that one."""
    if top >= len(SCHEDULES):
        return frozenset()
    kept = frozenset(s for s in cadences if SCHEDULES.index(s) >= top)
    return kept | {SCHEDULES[top]} if kept != cadences else kept


# ------------------------------------------------------------------------- helpers
_OPS: dict[str, Callable[[object, object], bool]] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def _version(text: str) -> tuple[int, ...] | None:
    m = re.fullmatch(r"(\d+(?:\.\d+)*)t?", text or "")
    return tuple(int(part) for part in m[1].split(".")) if m else None


def _probe(**fields: object) -> Any:
    """A stand-in run with just these fields, for an atom to test one value."""
    return types.SimpleNamespace(**fields)


def _get(env: Environment, field: str) -> Any:
    """A setting's value in a run; the harness and the device capability are derived."""
    if field == "harness":
        return _harness(env)
    return _capability(env) if field == "device_capability" else getattr(env, field)


def _family(name: str) -> str | None:
    families = [f for f, (prefix, _) in DEVICES.items() if name.startswith(prefix)]
    return families[0] if families else None


def _capability(env: Environment) -> str:
    family = _family(env.device_name)
    return DEVICES[family][1] if family else ""


def _harness(env: Environment) -> str | None:
    """The harness whose variables the run has, judged on all that harnesses set."""
    owned = {name for variables in HARNESSES.values() for name in variables}
    unset = ("", "0")
    actual = {n: env.flags[n] for n in owned if env.flags.get(n, "") not in unset}
    return next((h for h, variables in HARNESSES.items() if variables == actual), None)


def _operand_error(op: str, other: object) -> TypeError:
    hint = ""
    if isinstance(other, (int, str)):
        hint = (
            " -- comparisons bind looser than & and | in Python; parenthesize them,"
            " e.g. (tci.device_count >= 2) & tci.os.linux"
        )
    what = type(other).__name__
    return TypeError(
        f"unsupported operand type(s) for {op}: '{what}' and 'Selector'{hint}"
    )


def _flatten(items: Iterable[object]) -> Iterator[object]:
    for x in items:
        if isinstance(x, (list, tuple)):
            yield from _flatten(x)
        else:
            yield x


def _what(x: object) -> str:
    if isinstance(x, Cadence):
        return repr(x)
    return str(x) if isinstance(x, (Selector, Timed)) else type(x).__name__


def _platforms(
    items: tuple[object, ...], verb: str, timed: bool = True
) -> tuple[Selector | Timed, ...]:
    """A statement's platforms, lists flattened; X.when(...) only if timed is True."""
    out = tuple(_flatten(items))
    if not out:
        raise TypeError(f"{verb}() needs platforms")
    for x in out:
        if isinstance(x, Cadence):
            hint = "a platform's own cadence is written X.when(tci.periodic)"
            raise TypeError(f"{verb}() takes platforms, got the cadence {x!r}; {hint}")
        if isinstance(x, Timed) and not timed:
            hint = "the test's cadences go in their own decorator, @tci.when(...)"
            raise TypeError(f"{verb}() takes platforms without cadences; {hint}")
        if not isinstance(x, (Selector, Timed)):
            hint = ""
            if isinstance(x, Setting):
                hint = f" -- compare it, e.g. (tci.{x.name} >= ...)"
            what = type(x).__name__
            raise TypeError(f"{verb}() takes platform selectors, got {what}{hint}")
    return out  # type: ignore[return-value]


def _cadences(cadences: tuple[object, ...], verb: str) -> frozenset[str]:
    if not cadences:
        raise TypeError(f"{verb}() needs cadences, e.g. tci.trunk")
    for c in cadences:
        if not isinstance(c, Cadence):
            raise TypeError(f"{verb}() takes cadences, got {_what(c)}")
    return frozenset(c.name for c in cadences)  # type: ignore[attr-defined]


def _conj(a: Selector, b: Selector) -> Selector:
    if a == ANY or b == NOTHING:
        return b
    if b == ANY or a == NOTHING:
        return a
    return And(a, b)


def _disj(a: Selector, b: Selector) -> Selector:
    if a == ANY or b == ANY:
        return ANY
    return b if a == NOTHING else a if b == NOTHING else Or(a, b)


def _show_timed(sel: Selector, cadences: Iterable[str]) -> str:
    return f"{_show(sel, 4)}.when({', '.join(s for s in SCHEDULES if s in cadences)})"


def _show_pairs(sign: str, pairs: Mapping[str, Selector]) -> list[str]:
    """Pairs added or removed, one entry per selector with all of its cadences."""
    groups: dict[str, tuple[Selector, set[str]]] = {}
    for s, sel in pairs.items():
        if sel != NOTHING:
            groups.setdefault(str(sel), (sel, set()))[1].add(s)
    return [f"{sign} {_show_timed(sel, c)}" for sel, c in groups.values()]


def _show(sel: Selector, ctx: int) -> str:
    """The expression syntax specs print in. ctx is where sel appears: 0 at the top,
    1 inside |, 2 inside &, 3 inside ~, 4 before .when()."""

    def wrap(text: str, prec: int) -> str:
        return f"({text})" if prec < ctx else text

    def cmp(text: str) -> str:
        return f"({text})" if ctx > 0 else text

    def values(vs: Iterable[object]) -> str:
        return ", ".join(str(v) if isinstance(v, int) else f'"{v}"' for v in vs)

    if isinstance(sel, AnyRun):
        return "ANY"
    if isinstance(sel, Named):
        return sel.name
    if isinstance(sel, Or):
        return wrap(f"{_show(sel.a, 1)} | {_show(sel.b, 1)}", 1)
    if isinstance(sel, And):
        return wrap(f"{_show(sel.a, 2)} & {_show(sel.b, 2)}", 2)
    if isinstance(sel, Not):
        return wrap(f"~{_show(sel.a, 3)}", 3)
    if isinstance(sel, Is):
        if sel.one_of:
            return f"{sel.field}.one_of({values(sel.values)})"
        return f"{sel.field}.{sel.values[0]}"
    if isinstance(sel, Device):
        if sel.one_of:
            return f"device_name.one_of({values(sel.families)})"
        return f"device_name.{sel.families[0]}"
    if isinstance(sel, Compare):
        return cmp(f"{sel.field} {sel.op} {values([sel.value])}")
    if isinstance(sel, VersionOneOf):
        return f"{sel.field}.one_of({values(sel.values)})"
    if isinstance(sel, FreeThreaded):
        return "python_version.free_threaded"
    if isinstance(sel, EnvVar):
        return f"env_vars.{sel.name}"
    if isinstance(sel, EnvIs) and sel.one_of:
        return f"env_vars.{sel.name}.one_of({values(sel.values)})"
    if isinstance(sel, EnvIs):
        return cmp(f'env_vars.{sel.name} == "{sel.values[0]}"')
    raise TypeError(f"not a selector: {sel!r}")
