"""Where and when tests run, as data.

Decorators on TestCase classes and test functions state where a test runs and when, and
never skip anything. Every test starts from TestCase's: DEFAULT, below, on every PR and
every merge. This test runs on DEFAULT's platforms but Windows on every PR and every
merge, and on H100s periodically:

    @tci.exclude(tci.os.windows, reason="uses fork()")
    @tci.include(tci.device_name.h100.when(tci.periodic))
    class TestFoo(TestCase): ...

pytest --list-platforms (--list-platforms-json) prints where and when each collected
test runs (plugin.py). How tci works is in _core.py.
"""

from torch.testing._internal.tci._core import (
    accelerator,
    accelerator_version,
    ANY,
    ATTR,
    cc_compiler,
    cc_compiler_version,
    cpu_architecture,
    cpu_capability,
    describe,
    device_capability,
    device_count,
    device_name,
    env_vars,
    exclude,
    explain,
    harness,
    include,
    named,
    NOTHING,
    on,
    os,
    os_version,
    periodic,
    pull,
    python_version,
    resolve,
    runs_on,
    Selector,
    Spec,
    Statements,
    trunk,
    when,
)
from torch.testing._internal.tci.vocabulary import DEVICES, ENUMS, HARNESSES, SCHEDULES


__all__ = [
    # Decorators, read top to bottom: the test's cadences, the platforms it runs on,
    # platforms to add, platforms to leave out (with a reason, for the HUD), and
    # platforms to run at cadences of their own.
    "when",
    "runs_on",
    "include",
    "exclude",
    "on",
    # Cadences: every PR, every merge, the scheduler's timer. X.when(...) gives the
    # platforms in X cadences of their own.
    "pull",
    "trunk",
    "periodic",
    # A setting for each part of a platform, and the harness its tests run in: name a
    # value, tci.os.linux, or compare, tci.device_count >= 2, and combine with & | ~ -.
    "harness",
    "os",
    "os_version",
    "cpu_architecture",
    "cpu_capability",
    "accelerator",
    "accelerator_version",
    "device_name",
    "device_count",
    "device_capability",
    "python_version",
    "cc_compiler",
    "cc_compiler_version",
    "env_vars",
    # Named platforms: DEFAULT, GPU and MULTI_GPU, below; every platform and none; and
    # named(), to give a selector a name.
    "DEFAULT",
    "GPU",
    "MULTI_GPU",
    "ANY",
    "NOTHING",
    "named",
    # The values settings and cadences take (vocabulary.py).
    "ENUMS",
    "DEVICES",
    "HARNESSES",
    "SCHEDULES",
    # For tools: a test's statements folded into a Spec, and what it comes to.
    "resolve",
    "explain",
    "describe",
    "Selector",
    "Spec",
    "Statements",
    "ATTR",
]


# Where every test runs, on every PR and every merge (TestCase's when), unless it says
# otherwise: each kind of machine that runs the whole suite in pull and trunk today,
# with the Pythons it runs. Editing this changes the cost of CI; CI maintainers own it.
DEFAULT = named(
    "DEFAULT",
    harness.plain
    & (
        (
            os.linux
            & cpu_architecture.x86_64
            & cpu_capability.avx512
            & accelerator.cpu
            & python_version.one_of("3.11", "3.14", "3.14t")
        )
        | (
            os.linux
            & cpu_architecture.aarch64
            & cpu_capability.one_of("sve128", "sve256")
            & accelerator.cpu
            & python_version.one_of("3.11")
        )
        | (
            os.linux
            & cpu_architecture.x86_64
            & cpu_capability.avx2
            & accelerator.cuda
            & device_name.l4
            & (device_count == 1)
            & python_version.one_of("3.11")
        )
        | (
            os.linux
            & cpu_architecture.x86_64
            & cpu_capability.avx512
            & accelerator.rocm
            & device_name.mi350x
            & (device_count == 1)
            & python_version.one_of("3.11")
        )
        | (
            os.macos
            & cpu_architecture.aarch64
            & cpu_capability.default
            & accelerator.mps
            & device_name.m1
            & (device_count == 1)
            & python_version.one_of("3.12")
        )
        | (
            os.windows
            & cpu_architecture.x86_64
            & cpu_capability.avx512
            & accelerator.cpu
            & python_version.one_of("3.11")
        )
    )
    # the harnesses that run the whole suite, on the builds CI makes for them
    | (
        (harness.asan | harness.crossref)
        & os.linux
        & cpu_architecture.x86_64
        & cpu_capability.avx512
        & accelerator.cpu
        & python_version.one_of("3.11")
    )
    | (
        harness.dynamo
        & os.linux
        & cpu_architecture.x86_64
        & cpu_capability.avx512
        & accelerator.cpu
        & python_version.one_of("3.11", "3.12", "3.13", "3.14", "3.14t", "3.15")
    ),
)
# Platforms a test can ask for by name: tci.runs_on(tci.MULTI_GPU).
GPU = named(
    "GPU", (accelerator.cuda | accelerator.rocm | accelerator.xpu) & (device_count >= 1)
)
MULTI_GPU = named("MULTI_GPU", GPU & (device_count >= 2))
