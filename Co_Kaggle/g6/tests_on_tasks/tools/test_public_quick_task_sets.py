"""No-GPU integrity checks for nested public quick sets and gold-smoke evidence."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from materialize_quick_task_set import materialize

G6=Path(__file__).resolve().parents[2]
DATA=G6/"competition_material/competition_data/tasks.jsonl"
SETS=G6/"tests_on_tasks/quick_sets"


class QuickTaskSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids15=json.loads((SETS/"quick_15_ids.json").read_text())
        cls.ids30=json.loads((SETS/"quick_30_ids.json").read_text())
        cls.evidence=json.loads((SETS/"screening_evidence_20261027.json").read_text())
        cls.tasks=[json.loads(s) for s in DATA.read_text().splitlines() if s.strip()]
        cls.byid={r["instance_id"]:r for r in cls.tasks}

    def test_nested_and_unique(self):
        self.assertEqual(len(self.ids15),15)
        self.assertEqual(len(self.ids30),30)
        self.assertEqual(self.ids15,self.ids30[:15])
        self.assertEqual(len(set(self.ids30)),30)
        self.assertFalse(set(self.ids30)-set(self.byid))

    def test_all_selected_gold_verified_and_base_red(self):
        evidence={x["instance_id"]:x for x in self.evidence["results"]}
        self.assertEqual(self.evidence["screened"],57)
        self.assertEqual(self.evidence["passed"],37)
        self.assertEqual(len(evidence),57)
        for id in self.ids30:
            item=evidence[id]
            with self.subTest(id=id):
                self.assertEqual(item["status"],"RED_GREEN")
                self.assertIn(item["baseline_exit_code"],(1,2))
                self.assertEqual(item["gold_exit_code"],0)
                self.assertTrue(self.byid[id]["patch"].strip())
                self.assertTrue(self.byid[id]["test_patch"].strip())

    def test_distribution(self):
        for ids,distribution in [
            (self.ids15,{"Textualize/rich":12,"psf/requests":2,"fastapi/fastapi":1}),
            (self.ids30,{"Textualize/rich":24,"psf/requests":5,"fastapi/fastapi":1}),
        ]:
            self.assertEqual(dict(Counter(self.byid[id]["repo"] for id in ids)),distribution)
        self.assertNotIn("rich_3468",self.ids30)

    def test_materializer_preserves_full_source_rows(self):
        with tempfile.TemporaryDirectory() as dir:
            for n in (15,30):
                output=Path(dir)/f"set{n}.jsonl"
                materialize(DATA,SETS/f"quick_{n}_ids.json",output)
                rows=[json.loads(s) for s in output.read_text().splitlines() if s]
                ids=self.ids15 if n==15 else self.ids30
                self.assertEqual(rows,[self.byid[id] for id in ids])

    def test_report_covers_every_id(self):
        report=(SETS/"quick_task_sets_review.md").read_text()
        for id in self.ids30:
            self.assertIn("| "+chr(96)+id+chr(96)+" |",report)


if __name__=="__main__":
    unittest.main()
