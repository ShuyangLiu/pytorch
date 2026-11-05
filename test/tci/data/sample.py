# test_tci.py's TestPlugin collects this file with pytest; run_test.py doesn't run it.
from torch.testing._internal import tci
from torch.testing._internal.common_utils import TestCase


class TestSample(TestCase):
    def test_default(self):
        pass

    @tci.include(tci.harness.inductor)
    def test_inductor(self):
        pass

    @tci.when(tci.periodic)
    @tci.include(tci.harness.inductor)
    def test_inductor_periodic(self):
        pass

    @tci.include(tci.harness.inductor)
    @tci.exclude(tci.env_vars.TCI_SAMPLE_SWITCH, reason="a local switch")
    def test_switch(self):
        pass

    @tci.exclude(tci.harness.inductor, reason="not under Inductor")
    def test_not_inductor(self):
        pass
