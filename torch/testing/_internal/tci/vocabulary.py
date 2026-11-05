"""The values tci's settings and cadences take: tci.os.linux, tci.device_name.h100,
tci.harness.dynamo and the rest are generated from these, never written out."""

# The values torchci/environment.py captures for each identity column.
ENUMS: dict[str, tuple[str, ...]] = {
    "os": ("linux", "macos", "windows"),
    "cpu_architecture": ("x86_64", "aarch64", "s390x", "ppc64le"),
    "cpu_capability": (
        "default",
        "avx2",
        "avx512",
        "amx",
        "sve128",
        "sve256",
        "vsx",
        "zvector",
    ),
    "cc_compiler": ("gcc", "clang", "msvc"),
    "accelerator": ("cpu", "cuda", "rocm", "xpu", "mps"),
}
# Device families: the device_name prefix, and the capability ci.environments lacks.
DEVICES: dict[str, tuple[str, str]] = {
    "t4": ("Tesla T4", "7.5"),
    "a10g": ("NVIDIA A10G", "8.6"),
    "l4": ("NVIDIA L4", "8.9"),
    "h100": ("NVIDIA H100", "9.0"),
    "b200": ("NVIDIA B200", "10.0"),
    "gb300": ("NVIDIA GB300", "10.3"),
    "mi210": ("AMD Instinct MI210", "gfx90a"),
    "mi300x": ("AMD Instinct MI300X", "gfx942"),
    "mi350x": ("AMD Instinct MI350X", "gfx950"),
    "rx7900": ("AMD Radeon RX 7900", "gfx1100"),
    "m1": ("Apple M1", ""),
    "m2_pro": ("Apple M2 Pro", ""),
}
# When a run happens, most frequent first: every push to a PR, every merge to main, the
# scheduler's timer.
SCHEDULES = ("pull", "trunk", "periodic")
# The predefined ways to run the suite: the env vars each one sets. Every CI job runs in
# exactly one; test_tci.py checks that the report records each variable.
HARNESSES: dict[str, dict[str, str]] = {
    "plain": {},
    "dynamo": {"PYTORCH_TEST_WITH_DYNAMO": "1"},
    "inductor": {"PYTORCH_TEST_WITH_DYNAMO": "1", "PYTORCH_TEST_WITH_INDUCTOR": "1"},
    "crossref": {"PYTORCH_TEST_WITH_CROSSREF": "1"},
    "asan": {"PYTORCH_TEST_WITH_ASAN": "1", "PYTORCH_TEST_WITH_UBSAN": "1"},
    "slow_gradcheck": {
        "PYTORCH_TEST_WITH_SLOW_GRADCHECK": "1",
        "PYTORCH_TEST_CUDA_MEM_LEAK_CHECK": "1",
    },
    "expandable_segments": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    "cutedsl": {"OPINFO_RESTRICT_TO_DSL": "cutedsl"},
}
