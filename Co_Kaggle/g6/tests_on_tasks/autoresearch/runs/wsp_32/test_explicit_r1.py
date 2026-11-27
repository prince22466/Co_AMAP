"""Unit test for both rank-2 union-of-layers and rank-1 disjoint-layer explicit exports."""
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


class ExplicitFactorsR1R2Test(unittest.TestCase):
    def test_rank1_disjoint_export_and_overlap_guard(self):
        with zipfile.ZipFile(archive) as z:
            c = load(z.read("adapters/wsp_r1/adapter_model.safetensors"))
        gen = torch.Generator().manual_seed(42)
        row = torch.zeros(recipe.D_IN, dtype=torch.bfloat16)
        row[100:116] = torch.randn(16, generator=gen)
        col = torch.randn(recipe.D_OUT, generator=gen).bfloat16()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            disjoint = root / "disjoint.safetensors"
            save_file({"35.A": row, "35.B": col}, str(disjoint))
            for ctrl in ("candidate", "zero", "random"):
                m = recipe.compile_explicit(archive, disjoint, root / f"r1_{ctrl}", ctrl, rank=1)
                self.assertEqual(m["rank"], 1)
                self.assertEqual(m["active_layers"], [29, 31, 34, 35, 37, 38])
                new = load((root / f"r1_{ctrl}" / "adapter_model.safetensors").read_bytes())
                for layer in [29, 31, 34, 37, 38]:
                    p = recipe.KEY_PREFIX_FMT.format(layer=layer)
                    self.assertTrue(torch.equal(new[p + ".lora_A.weight"], c[p + ".lora_A.weight"]))
                    self.assertTrue(torch.equal(new[p + ".lora_B.weight"], c[p + ".lora_B.weight"]))
                p35 = recipe.KEY_PREFIX_FMT.format(layer=35)
                a35, b35 = new[p35 + ".lora_A.weight"], new[p35 + ".lora_B.weight"]
                self.assertEqual(a35.shape, (1, recipe.D_IN))
                self.assertEqual(b35.shape, (recipe.D_OUT, 1))
                if ctrl == "zero":
                    self.assertEqual(torch.count_nonzero(a35).item() + torch.count_nonzero(b35).item(), 0)
                elif ctrl == "candidate":
                    self.assertTrue(torch.equal(a35[0], row))
                    self.assertTrue(torch.equal(b35[:, 0], col))
            overlap = root / "overlap.safetensors"
            save_file({"29.A": row, "29.B": col}, str(overlap))
            with self.assertRaises(ValueError):
                recipe.compile_explicit(archive, overlap, root / "bad_r1", "candidate", rank=1)


if __name__ == "__main__":
    unittest.main()
