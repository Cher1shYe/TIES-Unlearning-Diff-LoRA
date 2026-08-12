import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


class FakeTensor:
    def __init__(self, values, shape=None):
        self.values = list(values)
        self.shape = shape or (len(self.values),)

    def abs(self):
        return FakeTensor([abs(value) for value in self.values], self.shape)

    def numel(self):
        return len(self.values)

    def view(self, *_shape):
        return FakeTensor(self.values, (len(self.values),))

    def reshape(self, *_shape):
        return FakeTensor(self.values, (len(self.values),))

    def reshape_as(self, other):
        return FakeTensor(self.values, other.shape)

    def __ge__(self, threshold):
        return FakeTensor([1.0 if value >= threshold else 0.0 for value in self.values], self.shape)

    def __setitem__(self, indices, value):
        if isinstance(indices, int):
            indices = [indices]
        for index in indices:
            self.values[index] = value

    def float(self):
        return self

    def nonzero_indices(self):
        return [index for index, value in enumerate(self.values) if value]


def load_ties_lora_with_fake_torch():
    torch_module = types.ModuleType("torch")
    nn_module = types.ModuleType("torch.nn")
    nn_module.Module = type("Module", (), {})
    nn_module.Linear = type("Linear", (), {})
    nn_module.Parameter = type("Parameter", (), {})
    torch_module.Tensor = FakeTensor
    torch_module.nn = nn_module
    torch_module.ones_like = lambda tensor: FakeTensor([1.0] * tensor.numel(), tensor.shape)
    torch_module.zeros_like = lambda tensor: FakeTensor([0.0] * tensor.numel(), tensor.shape)

    def topk(tensor, count):
        values = sorted(tensor.values, reverse=True)[:count]
        return SimpleNamespace(values=values)

    def argsort(tensor, descending=False, stable=False):
        self_key = (lambda index: (-tensor.values[index], index)) if descending else (lambda index: (tensor.values[index], index))
        return sorted(range(tensor.numel()), key=self_key)

    torch_module.topk = topk
    torch_module.argsort = argsort

    module_name = "ties_lora_trim_contract_under_test"
    module_path = ROOT / "models" / "ties_lora.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    previous_torch = sys.modules.get("torch")
    previous_nn = sys.modules.get("torch.nn")
    sys.modules["torch"] = torch_module
    sys.modules["torch.nn"] = nn_module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch
        if previous_nn is None:
            sys.modules.pop("torch.nn", None)
        else:
            sys.modules["torch.nn"] = previous_nn
    return module


class TiesTrimContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_ties_lora_with_fake_torch()

    def test_trim_keeps_exact_count_when_cutoff_magnitudes_tie(self):
        layer = SimpleNamespace(trim_ratio=0.2)
        delta = FakeTensor([0.0, 10.0, 1.0, 2.0, 9.0, -9.0, 9.0, 3.0, 4.0, 5.0])

        mask = self.module.TIESUnlearnLoRALinear._trim_mask(layer, delta)

        self.assertEqual([1, 4], mask.nonzero_indices())
        self.assertEqual(2, sum(mask.values))

    def test_non_positive_trim_ratio_is_rejected(self):
        for ratio in (0.0, -0.1):
            with self.subTest(ratio=ratio):
                with self.assertRaisesRegex(ValueError, "trim_ratio"):
                    self.module.TIESUnlearnLoRALinear._trim_mask(
                        SimpleNamespace(trim_ratio=ratio), FakeTensor([1.0, 2.0])
                    )

    def test_trim_ratio_one_keeps_every_element(self):
        mask = self.module.TIESUnlearnLoRALinear._trim_mask(
            SimpleNamespace(trim_ratio=1.0), FakeTensor([0.0, -2.0, 3.0])
        )
        self.assertEqual([0, 1, 2], mask.nonzero_indices())


if __name__ == "__main__":
    unittest.main()
