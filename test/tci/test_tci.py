# Owner(s): ["module: ci"]

import importlib.util
import json
import os
import re
import subprocess
import sys
import types
import unittest
from pathlib import Path
from typing import Any

from torch.testing._internal import tci
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
    TestEnvironment,
)
from torch.testing._internal.tci import _core
from torch.testing._internal.torchci.environment import capture, Environment


def platform(**fields: Any) -> Environment:
    """A Linux AVX512 CPU host on Python 3.11 in the plain harness, but for fields."""
    linux = {
        "os": "linux",
        "os_version": "22.04",
        "cpu_architecture": "x86_64",
        "cpu_capability": "avx512",
        "python_version": "3.11",
        "cc_compiler": "gcc",
        "cc_compiler_version": "11.4",
        "accelerator": "cpu",
        "accelerator_version": "",
        "device_count": 0,
        "device_name": "",
        "flags": {},
        "properties": {},
    }
    return Environment(**(linux | fields))


DYNAMO = {"PYTORCH_TEST_WITH_DYNAMO": "1"}
CUDA = {"accelerator": "cuda", "accelerator_version": "13.0"}
ROCM = {"accelerator": "rocm", "accelerator_version": "7.1"}
MAC = {
    "os": "macos",
    "os_version": "15.5",
    "cpu_architecture": "aarch64",
    "cpu_capability": "default",
    "cc_compiler": "clang",
    "cc_compiler_version": "17.0.0",
    "python_version": "3.12",
    "accelerator": "mps",
    "device_count": 1,
}
# A small catalog: DEFAULT's six platforms, then six it leaves out.
CATALOG = {
    "linux": platform(),
    "dynamo": platform(python_version="3.12", flags=DYNAMO),
    "windows": platform(
        os="windows",
        os_version="10.0.20348",
        cc_compiler="msvc",
        cc_compiler_version="194134120",
    ),
    "m1": platform(**MAC, device_name="Apple M1"),
    "l4": platform(
        **CUDA, cpu_capability="avx2", device_count=1, device_name="NVIDIA L4"
    ),
    "mi350x": platform(**ROCM, device_count=1, device_name="AMD Instinct MI350X VF"),
    "amx": platform(cpu_capability="amx"),
    "inductor": platform(flags=DYNAMO | {"PYTORCH_TEST_WITH_INDUCTOR": "1"}),
    "m2_pro": platform(**MAC, device_name="Apple M2 Pro"),
    "h100": platform(**CUDA, device_count=1, device_name="NVIDIA H100 80GB HBM3"),
    "b200x8": platform(**CUDA, device_count=8, device_name="NVIDIA B200"),
    "mi300x_x2": platform(**ROCM, device_count=2, device_name="AMD Instinct MI300X"),
}
DEFAULT = ("linux", "dynamo", "windows", "m1", "l4", "mi350x")
B200X8 = tci.device_name.b200 & (tci.device_count == 8)
H100X1 = tci.device_name.h100 & (tci.device_count == 1)


def names(selector: tci.Selector) -> str:
    return " ".join(name for name, env in CATALOG.items() if selector.matches(env))


def where(spec: tci.Spec) -> dict[str, str]:
    """The catalog's platforms the spec runs on, with their cadences."""
    out = {}
    for name, env in CATALOG.items():
        if cadences := spec.cadences_of(env):
            out[name] = " ".join(s for s in tci.SCHEDULES if s in cadences)
    return out


def at(cadences: str, *platforms: str) -> dict[str, str]:
    return dict.fromkeys(platforms, cadences)


def but(*platforms: str) -> list[str]:
    """DEFAULT's platforms, without these."""
    return [p for p in DEFAULT if p not in platforms]


def described(above: list[tci.Statements], below: list[tci.Statements]) -> str:
    # expecttest stores only ASCII, so the borders are drawn in it
    return "\n".join(tci.describe(fold(above, below), unicode=False))


def listed(spec: tci.Spec) -> dict[str, str]:
    """The catalog's platforms describe()'s rows name, with their cadences."""

    def matches(row: dict[str, Any], env: Environment) -> bool:
        family = _core._family(env.device_name)
        for key, values in row.items():
            if key.startswith("env_vars."):
                value = env.flags.get(key.removeprefix("env_vars."), "")
            else:
                value = family if key == "device_name" else _core._get(env, key)
            if not values.holds(key, value):
                return False
        return True

    out: dict[str, list[str]] = {}
    for s, region in _core._regions(spec).items():
        rows = _core._rows(region)
        for name, env in CATALOG.items():
            if any(matches(row, env) for row in rows):
                out.setdefault(name, []).append(s)
    return {name: " ".join(cadences) for name, cadences in out.items()}


def fold(above: list[tci.Statements], below: list[tci.Statements]) -> tci.Spec:
    """The spec of a test with the statements below, in a class with those above, each
    list in reading order."""

    def test_x(self):
        pass

    for statements in reversed(below):
        statements(test_x)
    cls = type("TestX", (TestCase,), {"test_x": test_x})
    for statements in reversed(above):
        statements(cls)
    return tci.resolve(cls, test_x)


# Where tci's own tests run: they are plain Python over tci's data, so DEFAULT's Linux
# CPU machines in the plain harness cover them, on each Python those have.
LINUX_CPU = tci.DEFAULT & tci.harness.plain & tci.os.linux & tci.accelerator.cpu


@tci.runs_on(LINUX_CPU)
class TestSelectors(TestCase):
    @parametrize(
        "selector, expected",
        [
            subtest((tci.DEFAULT, " ".join(DEFAULT)), name="default"),
            subtest((tci.GPU, "l4 mi350x h100 b200x8 mi300x_x2"), name="gpu"),
            subtest((tci.MULTI_GPU, "b200x8 mi300x_x2"), name="multi_gpu"),
            subtest((tci.harness.dynamo, "dynamo"), name="harness"),
            subtest(
                (tci.env_vars.PYTORCH_TEST_WITH_DYNAMO, "dynamo inductor"), name="var"
            ),
            subtest((tci.env_vars.TCI_UNSET, ""), name="unset_var"),
            subtest((tci.python_version >= "3.12", "dynamo m1 m2_pro"), name="version"),
            subtest((tci.device_capability >= "9.0", "h100 b200x8"), name="capability"),
            subtest((tci.device_capability == "gfx942", "mi300x_x2"), name="gfx"),
            subtest((tci.GPU - tci.accelerator.rocm, "l4 h100 b200x8"), name="minus"),
            subtest(
                (
                    tci.GPU.exclude(
                        [tci.accelerator.rocm, tci.device_name.h100], tci.device_name.l4
                    ),
                    "b200x8",
                ),
                name="exclude",
            ),
            subtest(
                (tci.device_name.one_of("h100", "b200"), "h100 b200x8"), name="one_of"
            ),
            subtest(
                (
                    tci.os.linux & tci.accelerator.cpu | tci.os.windows,
                    "linux dynamo windows amx inductor",
                ),
                name="precedence",
            ),
        ],
    )
    def test_matches(self, selector, expected):
        self.assertEqual(names(selector), expected)

    def test_versions_compare_per_component(self):
        self.assertFalse(
            (tci.python_version >= "3.10").matches(platform(python_version="3.9"))
        )
        self.assertTrue(
            (tci.python_version >= "3.10").matches(platform(python_version="3.12"))
        )
        cuda = platform(accelerator_version="10.1.0")
        self.assertTrue((tci.accelerator_version >= "10.1").matches(cuda))

    def test_free_threading(self):
        free = platform(python_version="3.14t")
        self.assertFalse(tci.python_version.one_of("3.14").matches(free))
        self.assertTrue(tci.python_version.one_of("3.14t").matches(free))
        self.assertTrue(tci.python_version.free_threaded.matches(free))

    def test_env_var_values(self):
        conf = tci.env_vars.PYTORCH_CUDA_ALLOC_CONF
        env = platform(flags={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        self.assertTrue((conf == "expandable_segments:True").matches(env))
        self.assertTrue(conf.one_of("expandable_segments:True", "unused").matches(env))
        self.assertFalse(conf.matches(env))
        self.assertTrue(tci.harness.expandable_segments.matches(env))

    def test_str(self):
        selector = (tci.DEFAULT | tci.harness.inductor) & ~tci.os.windows
        self.assertEqual(str(selector), "(DEFAULT | harness.inductor) & ~os.windows")
        self.assertEqual(
            str(B200X8.when(tci.periodic)),
            "(device_name.b200 & (device_count == 8)).when(periodic)",
        )
        self.assertEqual(
            str((~tci.os.windows).when(tci.pull)), "(~os.windows).when(pull)"
        )

    @parametrize(
        "make, error, message",
        [
            subtest(
                (
                    lambda: tci.device_count >= 2 & tci.os.linux,
                    TypeError,
                    "parenthesize them, e.g. (tci.device_count >= 2) & tci.os.linux",
                ),
                name="precedence",
            ),
            subtest(
                (
                    lambda: tci.os.linx,
                    AttributeError,
                    "tci.os has no value 'linx'; expected one of linux, macos, windows",
                ),
                name="unknown_value",
            ),
            subtest(
                (
                    lambda: tci.device_name.h200,
                    AttributeError,
                    "tci.device_name has no family 'h200'",
                ),
                name="unknown_family",
            ),
            subtest(
                (
                    lambda: tci.harness.dynamo & tci.harness.inductor,
                    ValueError,
                    "a run has exactly one harness; for either, write "
                    "tci.harness.dynamo | tci.harness.inductor",
                ),
                name="contradiction",
            ),
            subtest(
                (
                    lambda: tci.python_version >= 3.10,
                    TypeError,
                    "as a float, 3.10 would be 3.1",
                ),
                name="float_version",
            ),
            subtest(
                (
                    lambda: tci.env_vars.PYTORCH_TEST_WITH_DYNAMO == 1,
                    TypeError,
                    'e.g. (tci.env_vars.PYTORCH_TEST_WITH_DYNAMO == "1")',
                ),
                name="var_number",
            ),
            subtest(
                (
                    lambda: tci.os.linux and tci.os.windows,
                    TypeError,
                    "a selector has no truth value",
                ),
                name="and",
            ),
            subtest(
                (
                    lambda: tci.pull | tci.trunk,
                    TypeError,
                    "cadences do not combine with & or |: list them",
                ),
                name="combined_cadences",
            ),
            subtest(
                (
                    lambda: tci.os.linux & tci.os.macos.when(tci.trunk),
                    TypeError,
                    "combine platforms before giving them a cadence",
                ),
                name="cadence_on_part",
            ),
            subtest(
                (
                    lambda: tci.os.macos.when(tci.trunk).when(tci.pull),
                    TypeError,
                    "os.macos already has its cadences",
                ),
                name="two_whens",
            ),
            subtest(
                (
                    lambda: tci.when(tci.trunk).include(tci.os.linux),
                    TypeError,
                    "@tci.when(...) gives the whole test its cadences and stands alone",
                ),
                name="chain_after_when",
            ),
            subtest(
                (
                    lambda: tci.include(tci.os.linux).when(tci.pull),
                    TypeError,
                    "a chain takes no cadence",
                ),
                name="chained_when",
            ),
            subtest(
                (
                    lambda: tci.runs_on(tci.os.linux.when(tci.trunk)),
                    TypeError,
                    "runs_on() takes platforms without cadences",
                ),
                name="timed_runs_on",
            ),
            subtest(
                (
                    lambda: tci.include(tci.trunk),
                    TypeError,
                    "include() takes platforms, got the cadence tci.trunk",
                ),
                name="cadence_as_platform",
            ),
            subtest(
                (
                    lambda: tci.on(tci.os.linux),
                    TypeError,
                    "on() gives platforms their own cadences: "
                    "on(os.linux.when(tci.periodic))",
                ),
                name="on_without_cadence",
            ),
            subtest(
                (lambda: tci.include([]), TypeError, "include() needs platforms"),
                name="no_platforms",
            ),
            subtest(
                (
                    lambda: tci.include(tci.python_version),
                    TypeError,
                    "compare it, e.g. (tci.python_version >= ...)",
                ),
                name="setting",
            ),
            subtest(
                (
                    lambda: tci.exclude(tci.os.linux),
                    TypeError,
                    "missing 1 required keyword-only argument: 'reason'",
                ),
                name="no_reason",
            ),
        ],
    )
    def test_mistakes_raise(self, make, error, message):
        with self.assertRaisesRegex(error, re.escape(message)):
            make()


@tci.runs_on(LINUX_CPU)
class TestDecorators(TestCase):
    def test_statements_are_data(self):
        def test_x(self):
            pass

        self.assertIs(tci.include(tci.device_name.h100)(test_x), test_x)
        self.assertIs(tci.exclude(tci.os.windows, reason="uses fork()")(test_x), test_x)
        verbs = [statement.verb for statement in vars(test_x)[tci.ATTR]]
        self.assertEqual(verbs, ["exclude", "include"])

    def test_statements_stay_on_their_class(self):
        @tci.exclude(tci.os.windows, reason="uses fork()")
        class TestBase(TestCase):
            pass

        class TestSub(TestBase):
            pass

        self.assertNotIn(tci.ATTR, vars(TestSub))
        self.assertEqual(where(tci.resolve(TestSub)), at("pull trunk", *but("windows")))

    def test_chained_is_stacked(self):
        multi_gpu = tci.runs_on(tci.MULTI_GPU)
        nccl = tci.exclude(tci.accelerator.rocm, reason="NCCL")
        periodic = tci.on(tci.device_name.b200.when(tci.periodic))
        chain = multi_gpu.exclude(tci.accelerator.rocm, reason="NCCL").on(
            tci.device_name.b200.when(tci.periodic)
        )
        stacked = where(fold([], [multi_gpu, nccl, periodic]))
        self.assertEqual(stacked, at("periodic", "b200x8"))
        self.assertEqual(where(fold([], [chain])), stacked)


@tci.runs_on(LINUX_CPU)
class TestFold(TestCase):
    def test_root(self):
        self.assertEqual(where(fold([], [])), at("pull trunk", *DEFAULT))

    @parametrize(
        "cadences",
        [
            subtest((tci.periodic,), name="long"),
            subtest((tci.trunk,), name="slow"),
            subtest((tci.trunk, tci.periodic), name="merges_and_periodically"),
            subtest((tci.pull, tci.trunk, tci.periodic), name="also_periodically"),
        ],
    )
    def test_cadences(self, cadences):
        spec = fold([], [tci.when(*cadences)])
        self.assertEqual(where(spec), at(" ".join(c.name for c in cadences), *DEFAULT))

    @parametrize(
        "statements, expected",
        [
            subtest(
                (
                    [tci.exclude(tci.os.windows.when(tci.pull), reason="slow")],
                    at("pull trunk", *but("windows")) | at("trunk", "windows"),
                ),
                name="windows_on_merges",
            ),
            subtest(
                (
                    [tci.include(B200X8.when(tci.periodic))],
                    at("pull trunk", *DEFAULT) | at("periodic", "b200x8"),
                ),
                name="b200x8_periodically",
            ),
            subtest(
                (
                    [tci.on(tci.os.macos.when(tci.periodic))],
                    at("pull trunk", *but("m1")) | at("periodic", "m1"),
                ),
                name="macos_periodically",
            ),
            subtest(
                (
                    [tci.include(tci.device_name.m2_pro)],
                    at("pull trunk", *DEFAULT, "m2_pro"),
                ),
                name="m2_pro_on_prs",
            ),
            subtest(
                (
                    [
                        tci.include(
                            [
                                tci.device_name.h100,
                                tci.device_name.m2_pro.when(tci.periodic),
                            ]
                        )
                    ],
                    at("pull trunk", *DEFAULT, "h100") | at("periodic", "m2_pro"),
                ),
                name="list",
            ),
            subtest(
                (
                    [tci.when(tci.periodic), tci.runs_on(B200X8)],
                    at("periodic", "b200x8"),
                ),
                name="only_b200x8",
            ),
        ],
    )
    def test_one_level(self, statements, expected):
        self.assertEqual(where(fold([], statements)), expected)

    # The page's table of how statements above meet a level below.
    @parametrize(
        "above, below, expected",
        [
            subtest(
                (
                    [tci.on(tci.os.macos.when(tci.periodic))],
                    [tci.when(tci.trunk)],
                    at("trunk", *but("m1")) | at("periodic", "m1"),
                ),
                name="rarer_rule_stands",
            ),
            subtest(
                (
                    [tci.on(tci.accelerator.rocm.when(tci.trunk))],
                    [tci.when(tci.periodic)],
                    at("periodic", *DEFAULT),
                ),
                name="long_test_caps",
            ),
            subtest(
                (
                    [tci.on(tci.GPU.when(tci.trunk))],
                    [tci.when(tci.pull, tci.trunk)],
                    at("pull trunk", *but("l4", "mi350x"))
                    | at("trunk", "l4", "mi350x"),
                ),
                name="restating_changes_nothing",
            ),
            subtest(
                (
                    [tci.on(tci.device_name.l4.when(tci.pull, tci.trunk))],
                    [tci.when(tci.trunk)],
                    at("trunk", *DEFAULT),
                ),
                name="slow_test_caps",
            ),
            subtest(
                (
                    [tci.on(tci.GPU.when(tci.trunk))],
                    [tci.when(tci.pull, tci.trunk, tci.periodic)],
                    at("pull trunk periodic", *but("l4", "mi350x"))
                    | at("trunk periodic", "l4", "mi350x"),
                ),
                name="soak_reaches_every_platform",
            ),
            subtest(
                (
                    [tci.include(B200X8.when(tci.periodic))],
                    [tci.when(tci.trunk)],
                    at("trunk", *DEFAULT) | at("periodic", "b200x8"),
                ),
                name="pairs_carry_down",
            ),
            subtest(
                (
                    [tci.include(H100X1.when(tci.pull))],
                    [tci.when(tci.trunk)],
                    at("trunk", *DEFAULT, "h100"),
                ),
                name="pairs_are_capped",
            ),
            subtest(
                (
                    [tci.exclude(tci.GPU.when(tci.pull), reason="cost")],
                    [tci.include(tci.device_name.h100)],
                    at("pull trunk", *but("l4", "mi350x"))
                    | at("trunk", "l4", "mi350x", "h100"),
                ),
                name="removed_pairs_bind",
            ),
        ],
    )
    def test_rules_meet_levels_below(self, above, below, expected):
        self.assertEqual(where(fold(above, below)), expected)

    @parametrize(
        "statements, expected",
        [
            subtest(
                (
                    [
                        tci.exclude(tci.os.windows, reason="uses fork()"),
                        tci.include(tci.os.windows),
                    ],
                    at("pull trunk", *DEFAULT),
                ),
                name="later_include_wins",
            ),
            subtest(
                (
                    [
                        tci.include(tci.device_name.h100),
                        tci.exclude(tci.device_name.h100, reason="no quota"),
                    ],
                    at("pull trunk", *DEFAULT),
                ),
                name="later_exclude_wins",
            ),
            subtest(
                (
                    [
                        tci.on(
                            tci.GPU.when(tci.trunk),
                            tci.device_name.l4.when(tci.pull, tci.trunk),
                        )
                    ],
                    at("pull trunk", *but("mi350x")) | at("trunk", "mi350x"),
                ),
                name="left_to_right",
            ),
            subtest(
                (
                    [
                        tci.on(
                            tci.device_name.l4.when(tci.pull, tci.trunk),
                            tci.GPU.when(tci.trunk),
                        )
                    ],
                    at("pull trunk", *but("l4", "mi350x"))
                    | at("trunk", "l4", "mi350x"),
                ),
                name="later_on_wins",
            ),
        ],
    )
    def test_reading_order(self, statements, expected):
        self.assertEqual(where(fold([], statements)), expected)

    def test_deeper_level_wins(self):
        windows = tci.os.windows
        spec = fold([tci.exclude(windows, reason="fork")], [tci.include(windows)])
        self.assertEqual(where(spec), at("pull trunk", *DEFAULT))

    def test_mixins_fold_at_their_place_in_the_mro(self):
        @tci.when(tci.periodic)
        class LongMixin:
            pass

        @tci.on(tci.accelerator.rocm.when(tci.trunk))
        class RocmMerges(TestCase):
            pass

        class TestA(RocmMerges, LongMixin):
            pass

        class TestB(LongMixin, RocmMerges):
            pass

        for cls in (TestA, TestB):
            self.assertEqual(where(tci.resolve(cls)), at("periodic", *DEFAULT))

    def test_runs_on_drops_pairs_it_leaves_out(self):
        above = [tci.include(B200X8.when(tci.periodic))]
        spec = fold(above, [tci.exclude(tci.os.windows, reason="uses fork()")])
        self.assertEqual(
            where(spec), at("pull trunk", *but("windows")) | at("periodic", "b200x8")
        )
        spec = fold(above, [tci.runs_on(tci.device_name.h100)])
        self.assertEqual(where(spec), at("pull trunk", "h100"))

    def test_on_makes_paired_platforms_the_tests_own(self):
        above = [tci.include(B200X8.when(tci.periodic))]
        spec = fold(above, [tci.on(B200X8.when(tci.trunk))])
        want = at("pull trunk", *DEFAULT) | at("trunk", "b200x8")
        self.assertEqual(where(spec), want)

    def test_on_lifts_removed_pairs_on_the_tests_platforms(self):
        above = [tci.exclude(tci.GPU.when(tci.pull), reason="cost")]
        spec = fold(above, [tci.on(tci.device_name.l4.when(tci.pull, tci.trunk))])
        self.assertEqual(
            where(spec), at("pull trunk", *but("mi350x")) | at("trunk", "mi350x")
        )
        spec = fold(above, [tci.include(H100X1.when(tci.pull))])
        gpus = at("trunk", "l4", "mi350x")
        self.assertEqual(
            where(spec),
            at("pull trunk", *but("l4", "mi350x")) | gpus | at("pull", "h100"),
        )

    def test_root_without_cadences_runs_at_every_cadence(self):
        @tci.runs_on(tci.DEFAULT)
        class Root(unittest.TestCase):
            pass

        self.assertEqual(where(tci.resolve(Root)), at("pull trunk periodic", *DEFAULT))

    def test_one_runs_on_and_one_when_per_level(self):
        with self.assertRaisesRegex(TypeError, "TestX: at most one @tci.runs_on"):
            fold([tci.when(tci.periodic), tci.when(tci.trunk)], [])

    def test_needs_a_root(self):
        class Plain(unittest.TestCase):
            pass

        with self.assertRaisesRegex(TypeError, "Plain: no TestCase in its MRO has a"):
            tci.resolve(Plain)

    def test_spec_prints_as_expressions(self):
        spec = fold(
            [
                tci.on(tci.os.macos.when(tci.periodic)),
                tci.include(B200X8.when(tci.periodic)),
                tci.exclude(tci.os.windows.when(tci.pull), reason="slow"),
            ],
            [],
        )
        self.assertEqual(
            str(spec),
            "DEFAULT; when(pull, trunk); on(os.macos.when(periodic)); "
            "+ (device_name.b200 & (device_count == 8)).when(periodic); "
            "- os.windows.when(pull)",
        )


@tci.runs_on(LINUX_CPU)
class TestDescribe(TestCase):
    def test_default_spelled_out(self):
        self.assertExpectedInline(
            described([], []),
            """\
+--------------------------------------------------------------------------+
| When: every PR and every merge                                           |
+-----------+---------+----------------+-------------+---------------------+
| Harness   | OS      | CPU            | Accelerator | Python              |
+-----------+---------+----------------+-------------+---------------------+
| plain     | linux   | x86_64         | cpu         | 3.11, 3.14, 3.14t   |
|           |         | avx512         |             |                     |
|           +---------+----------------+-------------+---------------------+
|           | linux   | aarch64        | cpu         | 3.11                |
|           |         | sve128, sve256 |             |                     |
|           +---------+----------------+-------------+---------------------+
|           | linux   | x86_64         | cuda        | 3.11                |
|           |         | avx2           | l4          |                     |
|           |         |                | 1 device    |                     |
|           +---------+----------------+-------------+---------------------+
|           | linux   | x86_64         | rocm        | 3.11                |
|           |         | avx512         | mi350x      |                     |
|           |         |                | 1 device    |                     |
|           +---------+----------------+-------------+---------------------+
|           | macos   | aarch64        | mps         | 3.12                |
|           |         | default        | m1          |                     |
|           |         |                | 1 device    |                     |
|           +---------+----------------+-------------+---------------------+
|           | windows | x86_64         | cpu         | 3.11                |
|           |         | avx512         |             |                     |
+-----------+---------+----------------+-------------+---------------------+
| crossref, | linux   | x86_64         | cpu         | 3.11                |
|   asan    |         | avx512         |             |                     |
+-----------+---------+----------------+-------------+---------------------+
| dynamo    | linux   | x86_64         | cpu         | 3.11, 3.12, 3.13,   |
|           |         | avx512         |             |   3.14, 3.14t, 3.15 |
+-----------+---------+----------------+-------------+---------------------+""",
        )

    def test_excludes_narrow_rows(self):
        below = [
            tci.exclude(tci.accelerator.rocm, reason="NCCL"),
            tci.exclude(tci.device_count == 4, reason="flaky on 4"),
        ]
        self.assertExpectedInline(
            described([tci.runs_on(tci.MULTI_GPU)], below),
            """\
+-------------------------------------------------+
| When: every PR and every merge                  |
+---------+-----+-----+------------------+--------+
| Harness | OS  | CPU | Accelerator      | Python |
+---------+-----+-----+------------------+--------+
| any     | any | any | cuda, xpu        | any    |
|         |     |     | 2, 3, 5+ devices |        |
+---------+-----+-----+------------------+--------+
Excluded accelerator.rocm: NCCL
Excluded device_count == 4: flaky on 4""",
        )

    def test_a_table_per_cadence(self):
        above = [tci.runs_on(tci.os.linux | tci.os.macos)]
        below = [
            tci.on(tci.os.macos.when(tci.periodic)),
            tci.include(B200X8.when(tci.periodic)),
        ]
        self.assertExpectedInline(
            described(above, below),
            """\
+----------------------------------------------+
| When: every PR and every merge               |
+---------+-------+-----+-------------+--------+
| Harness | OS    | CPU | Accelerator | Python |
+---------+-------+-----+-------------+--------+
| any     | linux | any | any         | any    |
+---------+-------+-----+-------------+--------+
+----------------------------------------------+
| When: periodically                           |
+---------+-------+-----+-------------+--------+
| Harness | OS    | CPU | Accelerator | Python |
+---------+-------+-----+-------------+--------+
| any     | macos | any | any         | any    |
|         +-------+-----+-------------+--------+
|         | any   | any | b200        | any    |
|         |       |     | 8 devices   |        |
+---------+-------+-----+-------------+--------+""",
        )

    def test_capabilities_name_device_families(self):
        above = [tci.runs_on(tci.GPU & (tci.device_capability >= "9.0"))]
        self.assertExpectedInline(
            described(above, []),
            """\
+--------------------------------------------------+
| When: every PR and every merge                   |
+---------+-----+-----+-------------------+--------+
| Harness | OS  | CPU | Accelerator       | Python |
+---------+-----+-----+-------------------+--------+
| any     | any | any | cuda, rocm, xpu   | any    |
|         |     |     | h100, b200, gb300 |        |
|         |     |     | 1+ devices        |        |
+---------+-----+-----+-------------------+--------+""",
        )

    def test_versions(self):
        versions = tci.python_version.one_of("3.11", "3.14", "3.14t")
        above = [tci.runs_on(versions | (tci.python_version >= "3.15"))]
        below = [tci.exclude(tci.python_version.free_threaded, reason="needs the GIL")]
        self.assertExpectedInline(
            described(above, below),
            """\
+-------------------------------------------------------+
| When: every PR and every merge                        |
+---------+-----+-----+-------------+-------------------+
| Harness | OS  | CPU | Accelerator | Python            |
+---------+-----+-----+-------------+-------------------+
| any     | any | any | any         | 3.11, 3.14        |
|         +-----+-----+-------------+-------------------+
|         | any | any | any         | >= 3.15           |
|         |     |     |             | not free-threaded |
+---------+-----+-----+-------------+-------------------+
Excluded python_version.free_threaded: needs the GIL""",
        )

    def test_env_vars(self):
        below = [
            tci.exclude(tci.env_vars.TCI_SWITCH, reason="a switch"),
            tci.include(tci.env_vars.TCI_MODE.one_of("a", "")),
        ]
        self.assertExpectedInline(
            described([tci.runs_on(tci.os.linux)], below),
            """\
+-----------------------------------------------------------------+
| When: every PR and every merge                                  |
+---------+-------+-----+-------------+--------+------------------+
| Harness | OS    | CPU | Accelerator | Python | Env              |
+---------+-------+-----+-------------+--------+------------------+
| any     | linux | any | any         | any    | TCI_SWITCH not 1 |
|         +-------+-----+-------------+--------+------------------+
|         | any   | any | any         | any    | TCI_MODE="", a   |
+---------+-------+-----+-------------+--------+------------------+
Excluded env_vars.TCI_SWITCH: a switch""",
        )

    def test_harnesses_set_env_vars(self):
        # dynamo sets PYTORCH_TEST_WITH_DYNAMO and the other harnesses leave it unset,
        # so excluding it drops dynamo, and the top says it for the row of any harness
        above = [
            tci.include(tci.os.macos),
            tci.exclude(tci.env_vars.PYTORCH_TEST_WITH_DYNAMO, reason="nested"),
        ]
        self.assertExpectedInline(
            described(above, []),
            """\
+-----------------------------------------------------------------------------+
| When: every PR and every merge                                              |
| Env:  PYTORCH_TEST_WITH_DYNAMO not 1                                        |
+----------------+---------+----------------+-------------+-------------------+
| Harness        | OS      | CPU            | Accelerator | Python            |
+----------------+---------+----------------+-------------+-------------------+
| plain          | linux   | x86_64         | cpu         | 3.11, 3.14, 3.14t |
|                |         | avx512         |             |                   |
|                +---------+----------------+-------------+-------------------+
|                | linux   | aarch64        | cpu         | 3.11              |
|                |         | sve128, sve256 |             |                   |
|                +---------+----------------+-------------+-------------------+
|                | linux   | x86_64         | cuda        | 3.11              |
|                |         | avx2           | l4          |                   |
|                |         |                | 1 device    |                   |
|                +---------+----------------+-------------+-------------------+
|                | linux   | x86_64         | rocm        | 3.11              |
|                |         | avx512         | mi350x      |                   |
|                |         |                | 1 device    |                   |
|                +---------+----------------+-------------+-------------------+
|                | windows | x86_64         | cpu         | 3.11              |
|                |         | avx512         |             |                   |
+----------------+---------+----------------+-------------+-------------------+
| crossref, asan | linux   | x86_64         | cpu         | 3.11              |
|                |         | avx512         |             |                   |
+----------------+---------+----------------+-------------+-------------------+
| any            | macos   | any            | any         | any               |
+----------------+---------+----------------+-------------+-------------------+
Excluded env_vars.PYTORCH_TEST_WITH_DYNAMO: nested""",
        )

    def test_explain(self):
        below = [
            tci.exclude(tci.accelerator.rocm, reason="NCCL"),
            tci.exclude(tci.python_version >= "3.14", reason="not yet"),
            tci.include(B200X8.when(tci.periodic)),
        ]
        explained = tci.explain(fold([tci.runs_on(tci.MULTI_GPU)], below))
        on_prs = {
            "python_version": {"checks": ["not >= 3.14"]},
            "accelerator": ["cuda", "xpu"],
            "device_count": {"not": [0, 1]},
        }
        periodically = {"device_name": ["b200"], "device_count": [8]}
        excluded = {"accelerator.rocm": "NCCL", 'python_version >= "3.14"': "not yet"}
        expected = {
            "runs": [
                {"when": ["pull", "trunk"], "on": [on_prs]},
                {"when": ["periodic"], "on": [periodically]},
            ],
            "excluded": excluded,
        }
        self.assertEqual(explained, expected)

    def test_lines_fit_the_width(self):
        nvidia = tci.device_name.one_of("t4", "a10g", "l4", "h100", "b200", "gb300")
        reason = "the NVLink topology differs from every other machine's"
        multi_h100 = tci.device_name.h100 & (tci.device_count >= 2)
        spec = fold([tci.runs_on(nvidia)], [tci.exclude(multi_h100, reason=reason)])
        lines = tci.describe(spec, width=48, unicode=False)
        self.assertLessEqual(max(map(len, lines)), 48)
        self.assertExpectedInline(
            "\n".join(lines),
            """\
+----------------------------------------------+
| When: every PR and every merge               |
+---------+-----+-----+---------------+--------+
| Harness | OS  | CPU | Accelerator   | Python |
+---------+-----+-----+---------------+--------+
| any     | any | any | t4, a10g, l4, | any    |
|         |     |     |   b200, gb300 |        |
|         +-----+-----+---------------+--------+
|         | any | any | t4, a10g, l4, | any    |
|         |     |     |   h100, b200, |        |
|         |     |     |   gb300       |        |
|         |     |     | 0, 1 devices  |        |
+---------+-----+-----+---------------+--------+
Excluded device_name.h100 & (device_count >= 2):
  the NVLink topology differs from every other
  machine's""",
        )

    def test_cells_wrap_in_their_rows(self):
        lines = tci.describe(fold([], []), width=60, unicode=False)
        self.assertExpectedInline(
            "\n".join(lines),
            """\
+---------------------------------------------------------+
| When: every PR and every merge                          |
+-----------+---------+----------+-------------+----------+
| Harness   | OS      | CPU      | Accelerator | Python   |
+-----------+---------+----------+-------------+----------+
| plain     | linux   | x86_64   | cpu         | 3.11,    |
|           |         | avx512   |             |   3.14,  |
|           |         |          |             |   3.14t  |
|           +---------+----------+-------------+----------+
|           | linux   | aarch64  | cpu         | 3.11     |
|           |         | sve128,  |             |          |
|           |         |   sve256 |             |          |
|           +---------+----------+-------------+----------+
|           | linux   | x86_64   | cuda        | 3.11     |
|           |         | avx2     | l4          |          |
|           |         |          | 1 device    |          |
|           +---------+----------+-------------+----------+
|           | linux   | x86_64   | rocm        | 3.11     |
|           |         | avx512   | mi350x      |          |
|           |         |          | 1 device    |          |
|           +---------+----------+-------------+----------+
|           | macos   | aarch64  | mps         | 3.12     |
|           |         | default  | m1          |          |
|           |         |          | 1 device    |          |
|           +---------+----------+-------------+----------+
|           | windows | x86_64   | cpu         | 3.11     |
|           |         | avx512   |             |          |
+-----------+---------+----------+-------------+----------+
| crossref, | linux   | x86_64   | cpu         | 3.11     |
|   asan    |         | avx512   |             |          |
+-----------+---------+----------+-------------+----------+
| dynamo    | linux   | x86_64   | cpu         | 3.11,    |
|           |         | avx512   |             |   3.12,  |
|           |         |          |             |   3.13,  |
|           |         |          |             |   3.14,  |
|           |         |          |             |   3.14t, |
|           |         |          |             |   3.15   |
+-----------+---------+----------+-------------+----------+""",
        )

    def test_box_drawing(self):
        lines = tci.describe(
            fold([tci.runs_on(tci.os.linux | tci.accelerator.cuda)], [])
        )
        box = [
            "┌──────────────────────────────────────────────┐",
            "│ When: every PR and every merge               │",
            "├─────────┬───────┬─────┬─────────────┬────────┤",
            "│ Harness │ OS    │ CPU │ Accelerator │ Python │",
            "├─────────┼───────┼─────┼─────────────┼────────┤",
            "│ any     │ linux │ any │ any         │ any    │",
            "│         ├───────┼─────┼─────────────┼────────┤",
            "│         │ any   │ any │ cuda        │ any    │",
            "└─────────┴───────┴─────┴─────────────┴────────┘",
        ]
        self.assertEqual(lines, box)

    def test_runs_anywhere(self):
        below = [tci.include(tci.os.linux.when(tci.periodic))]
        self.assertExpectedInline(
            described([tci.runs_on(tci.ANY)], below),
            """\
+---------------------------------+
| When:  every PR and every merge |
| Where: any platform             |
+---------------------------------+
+----------------------------------------------+
| When: periodically                           |
+---------+-------+-----+-------------+--------+
| Harness | OS    | CPU | Accelerator | Python |
+---------+-------+-----+-------------+--------+
| any     | linux | any | any         | any    |
+---------+-------+-----+-------------+--------+""",
        )

    def test_runs_nowhere(self):
        below = [tci.exclude(tci.os.linux, reason="not yet")]
        self.assertExpectedInline(
            described([tci.runs_on(tci.os.linux)], below),
            """\
Runs on no platform.
Excluded os.linux: not yet""",
        )

    @parametrize(
        "above, below",
        [
            subtest(([], []), name="root"),
            subtest(([], [tci.when(tci.trunk, tci.periodic)]), name="soak"),
            subtest(
                ([tci.on(tci.os.macos.when(tci.periodic))], [tci.when(tci.trunk)]),
                name="retimed",
            ),
            subtest(
                (
                    [tci.on(tci.accelerator.rocm.when(tci.trunk))],
                    [tci.when(tci.periodic)],
                ),
                name="capped",
            ),
            subtest(
                (
                    [tci.exclude(tci.GPU.when(tci.pull), reason="cost")],
                    [tci.include(tci.device_name.h100, B200X8.when(tci.periodic))],
                ),
                name="pairs",
            ),
            subtest(
                (
                    [tci.runs_on(tci.MULTI_GPU | tci.os.windows)],
                    [tci.exclude(tci.device_capability < "9.0", reason="old")],
                ),
                name="capability",
            ),
            subtest(
                (
                    [tci.exclude(tci.MULTI_GPU - tci.accelerator.rocm, reason="r")],
                    [tci.on((tci.harness.dynamo | tci.os.macos).when(tci.periodic))],
                ),
                name="complements",
            ),
            subtest(
                (
                    [tci.exclude(tci.env_vars.PYTORCH_TEST_WITH_DYNAMO, reason="r")],
                    [tci.include(tci.harness.plain & tci.os.macos)],
                ),
                name="harness_env_vars",
            ),
        ],
    )
    def test_rows_name_what_runs(self, above, below):
        spec = fold(above, below)
        self.assertEqual(listed(spec), where(spec))


@tci.runs_on(LINUX_CPU)
class TestGenerators(TestCase):
    def test_parametrized_variants_keep_their_statements(self):
        class TestP(TestCase):
            @tci.exclude(tci.os.windows, reason="uses fork()")
            @parametrize("x", [1, 2])
            def test_p(self, x):
                pass

        instantiate_parametrized_tests(TestP)
        for name in ("test_p_x_1", "test_p_x_2"):
            spec = tci.resolve(TestP, getattr(TestP, name))
            self.assertEqual(where(spec), at("pull trunk", *but("windows")))

    def test_device_type_classes_fold_class_statements_once(self):
        @tci.on(tci.os.macos.when(tci.periodic))
        class TestD(TestCase):
            @tci.exclude(tci.os.windows, reason="uses fork()")
            def test_d(self, device):
                pass

        scope = {"TestD": TestD}
        instantiate_device_type_tests(TestD, scope, allow_mps=True, allow_xpu=True)
        self.assertTrue(scope)
        want = "DEFAULT & ~os.windows; when(pull, trunk); on(os.macos.when(periodic))"
        for cls in scope.values():
            test = getattr(cls, f"test_d_{cls.device_type}")
            self.assertEqual(str(tci.resolve(cls, test)), want)


@tci.runs_on(LINUX_CPU)
class TestEnvironmentSync(TestCase):
    def test_settings_match_the_environment(self):
        settings = {n for n, x in vars(tci).items() if isinstance(x, _core.Setting)}
        derived = {"device_capability", "harness"}
        self.assertEqual(settings - derived, set(CATALOG["linux"].identity()))

    def test_harness_variables_are_recorded(self):
        variables = {v for harness in tci.HARNESSES.values() for v in harness}
        self.assertEqual(variables - set(TestEnvironment.env_var_values), set())

    # every kind of machine the suite runs on must capture values tci knows
    @tci.runs_on(tci.DEFAULT & tci.harness.plain)
    def test_this_machine_has_atoms(self):
        env = capture()
        for setting, values in tci.ENUMS.items():
            self.assertIn(getattr(env, setting), values)

    def test_all_lists_the_public_api(self):
        modules = {n for n, x in vars(tci).items() if isinstance(x, types.ModuleType)}
        public = {n for n in vars(tci) if n[0] != "_"} - modules
        self.assertEqual(set(tci.__all__), public)


_TEST_DIR = Path(__file__).resolve().parent
_SAMPLE = _TEST_DIR / "data/sample.py"
# Drop the job's PYTEST_ADDOPTS and a PYTHONPATH whose repo root would shadow the
# installed torch, and don't write bytecode into test/.
_CHILD_ENV = {
    k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTEST_ADDOPTS")
} | {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}


# pytest --list-platforms in a subprocess: what it prints depends on the OS (paths, a
# console that can't encode box drawing), not on the device or the harness.
@tci.runs_on(tci.DEFAULT & tci.harness.plain & ~tci.GPU)
class TestPlugin(TestCase):
    def pytest(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(_SAMPLE),
                "-p",
                "no:cacheprovider",
                "-q",
            ]
            + list(args),
            cwd=_TEST_DIR,
            env=_CHILD_ENV | env,
            capture_output=True,
            encoding="utf-8",
            timeout=300,
        )

    def specs(self) -> dict[str, tci.Spec]:
        """The sample's specs, by test name."""
        spec = importlib.util.spec_from_file_location("tci_sample", _SAMPLE)
        sample = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sample)
        tests = vars(sample.TestSample).items()
        return {
            t: tci.resolve(sample.TestSample, f) for t, f in tests if t[:5] == "test_"
        }

    def test_list_platforms(self):
        proc = self.pytest("--list-platforms")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        # After a legend, each group of tests: their names, then their description.
        listed = {}
        for group in proc.stdout.split("\n\n")[1:]:
            lines = group.splitlines()
            described = [line[2:] for line in lines if line.startswith("  ")]
            for line in lines:
                if not line.startswith(" "):
                    listed[line.rsplit("::", 1)[1]] = described
        expected = {t: tci.describe(spec, width=78) for t, spec in self.specs().items()}
        self.assertEqual(listed, expected)
        self.assertLessEqual(max(map(len, proc.stdout.splitlines())), 80)

    def test_list_platforms_ascii(self):
        # a stdout that can't encode box drawing, as a Windows code page can't
        listing = self.pytest("--list-platforms", PYTHONIOENCODING="ascii").stdout
        self.assertIn("+---", listing)
        self.assertEqual(listing, listing.encode("ascii").decode())

    def test_list_platforms_json(self):
        proc = self.pytest("--list-platforms-json")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        listed = {}
        for group in json.loads(proc.stdout):
            for nodeid in group.pop("tests"):
                listed[nodeid.rsplit("::", 1)[1]] = group
        self.assertEqual(listed, {t: tci.explain(s) for t, s in self.specs().items()})

    def test_collection_errors_fail_the_listing(self):
        broken = str(_TEST_DIR / "data/broken.py")
        proc = self.pytest("--list-platforms-json", broken)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertEqual(len(json.loads(proc.stdout)), len(self.specs()))


instantiate_parametrized_tests(TestSelectors)
instantiate_parametrized_tests(TestFold)
instantiate_parametrized_tests(TestDescribe)


if __name__ == "__main__":
    run_tests()
