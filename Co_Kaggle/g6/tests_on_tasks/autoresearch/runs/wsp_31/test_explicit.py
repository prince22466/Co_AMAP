"""Regression test for union-of-layers packing and the serialized sparse write."""
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch
from safetensors.torch import load, save_file

research = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("wsp_recipe", research / "recipe.py")
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)
archive = research.parents[1] / "submission_nb/artifacts/submission_40.zip"


class ExplicitFactorsTest(unittest.TestCase):
    def test_overlapping_and_new_layers(self):
        with zipfile.ZipFile(archive) as z:
            c = load(z.read("adapters/wsp_r1/adapter_model.safetensors"))
        generator = torch.Generator().manual_seed(31)
        row = torch.zeros(recipe.D_IN, dtype=torch.bfloat16)
        row[5:21] = torch.randn(16, generator=generator)
        col = torch.randn(recipe.D_OUT, generator=generator).bfloat16()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            factors = root / "factors.safetensors"
            save_file({"17.A":row,"17.B":col,"29.A":-row,"29.B":col.clone()}, str(factors))
            for control in ("candidate", "random", "zero"):
                manifest = recipe.compile_explicit(archive, factors, root / control, control)
                self.assertEqual(manifest["active_layers"], [17,29,31,34,37,38])
                new = load((root / control / "adapter_model.safetensors").read_bytes())
                for layer in manifest["active_layers"]:
                    prefix = recipe.KEY_PREFIX_FMT.format(layer=layer)
                    ak, bk = prefix+".lora_A.weight", prefix+".lora_B.weight"
                    a, b = new[ak], new[bk]
                    self.assertTrue(torch.equal(a[:1], c.get(ak,torch.zeros_like(a[:1]))))
                    self.assertTrue(torch.equal(b[:,:1], c.get(bk,torch.zeros_like(b[:,:1]))))
                    if layer not in (17,29) or control == "zero":
                        self.assertEqual(torch.count_nonzero(a[1]).item()+torch.count_nonzero(b[:,1]).item(),0)
                    elif control == "candidate":
                        expected = row if layer == 17 else -row
                        self.assertTrue(torch.equal(expected, a[1]))
                        self.assertTrue(torch.equal(col, b[:,1]))
                        h = torch.randn(4,recipe.D_IN,generator=generator)
                        self.assertTrue(torch.allclose((h @ expected.float())[:,None]*col.float(),
                                                       (h @ a[1].float())[:,None]*b[:,1].float()))
                    else:
                        self.assertTrue(torch.equal(a[1].abs().sort().values, row.abs().sort().values))
                        self.assertTrue(torch.equal(b[:,1].abs().sort().values,col.abs().sort().values))
                        self.assertFalse(torch.equal(a[1].bool(),row.bool()))


if __name__ == "__main__":
    unittest.main()
