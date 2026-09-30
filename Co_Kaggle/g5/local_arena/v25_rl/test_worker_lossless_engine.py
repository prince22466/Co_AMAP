"""Boundary checks against the real Kaggriculture engine in requirements.txt."""
import copy
import random
import unittest

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from test_worker_lossless_harvest import crop, _OngoingExecutor
from test_worker_reward_contract import EXECUTOR, _obs
from worker_reward import compute_worker_reward


class EngineDecayTest(unittest.TestCase):
    def transition(self,tile,step,action):
        before=_obs(tile,day=step//24,hour=step%24)
        # The engine uses a square board. This test only works on tile (0,0).
        before["farms"][0]["tiles"]=[[None for _ in range(5)] for _ in range(5)]
        before["farms"][0]["tiles"][0][0]=tile
        after=copy.deepcopy(before)
        engine._apply_unit_action(after["farms"][0],after["private"],0,action,5,step//24,24,100)
        engine._decay_plants(after["farms"][0],step)
        after.update(day=(step+1)//24,hour=(step+1)%24)
        executor=_OngoingExecutor() if tile["crop"]=="TOMATO" else EXECUTOR
        result=compute_worker_reward(executor,before,{"farmer":action,"hands":[]},after)
        return after,result

    def test_harvest_at_first_decay_step_executes_before_decay(self):
        after,result=self.transition(crop(),120,["HARVEST"])
        self.assertIsNone(after["farms"][0]["tiles"][0][0])
        self.assertEqual(result.crop_units_harvested_total,4)
        self.assertEqual(result.lost_harvestable_units,0)

    def test_engine_gradual_decay_is_measured(self):
        after,result=self.transition(crop(),120,["PASS"])
        self.assertEqual(after["farms"][0]["tiles"][0][0]["yield_units"],3)
        self.assertEqual(result.crop_units_lost_to_decay,1)

    def test_engine_final_harvest_requires_cleanup_before_expiry(self):
        after,result=self.transition(crop(name="TOMATO",start=288),287,["HARVEST"])
        self.assertEqual(result.crops_to_weed,0)
        empty=after["farms"][0]["tiles"][0][0]
        self.assertEqual(empty["yield_units"],0)
        after,result=self.transition(empty,288,["DIG"])
        self.assertIsNone(after["farms"][0]["tiles"][0][0])
        self.assertEqual(result.crops_to_weed,0)
        self.assertEqual(result.crops_died,0)

    def test_engine_harvest_at_ongoing_expiry_saves_units_but_creates_weed(self):
        _,result=self.transition(crop(name="TOMATO",start=288),288,["HARVEST"])
        self.assertEqual(result.crop_units_harvested_total,4)
        self.assertEqual(result.lost_harvestable_units,0)
        self.assertEqual(result.crops_to_weed,1)

    def test_random_weed_after_successful_one_time_harvest_is_not_crop_loss(self):
        before=_obs(crop(),day=4,hour=23)
        after=copy.deepcopy(before)
        after["farms"][0]["tiles"]=[[None for _ in range(5)] for _ in range(5)]
        after["farms"][0]["tiles"][0][0]=copy.deepcopy(before["farms"][0]["tiles"][0][0])
        engine._apply_unit_action(after["farms"][0],after["private"],0,["HARVEST"],5,4,24,100)
        engine._spawn_weeds(after["farms"][0],5,1.0,random.Random(0))
        after.update(day=5,hour=0)
        result=compute_worker_reward(EXECUTOR,before,{"farmer":["HARVEST"],"hands":[]},after)
        self.assertEqual(after["farms"][0]["tiles"][0][0],{"kind":"WEED"})
        self.assertEqual(result.crop_units_harvested_total,4)
        self.assertEqual(result.crops_to_weed,0)
        self.assertEqual(result.lost_harvestable_units,0)


if __name__=="__main__":
    unittest.main()
