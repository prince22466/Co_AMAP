"""Planner animal setup priority, shared maintenance capacity and PASS masking."""
import copy
import unittest

import torch

from test_worker_route_commitment import _Executor, _PassLovingModel, _PlantLovingModel, _obs
from worker_policy import CANDIDATE_FEATURE_NAMES, Task, WorkerPolicy


def observation(worker_x=0, hour=5):
    obs=_obs(worker_x=worker_x,fed=True,wheat=0)
    obs["hour"]=hour
    obs["farms"][0]["tiles"][0][3]=None
    obs["private"]["shed"]={"WHEAT":20,"COW":1}
    obs["private"]["seeds"]={"WHEAT":2}
    return obs


def policy(model=None, collect=False):
    return WorkerPolicy(_Executor(),model or _PlantLovingModel(),torch.device("cpu"),
                        deterministic=True,collect=collect)


class AnimalPlacementTest(unittest.TestCase):
    def test_build_precedes_ordinary_planting(self):
        obs=observation()
        self.assertEqual(policy().unit_actions(obs,{(0,0):"COW"},{(1,0):"WHEAT"}),
                         [["BUILD_PASTURE"]])

    def test_pickup_precedes_ordinary_planting(self):
        obs=observation()
        obs["farms"][0]["tiles"][0][2]={"kind":"PASTURE"}
        self.assertEqual(policy().unit_actions(obs,{(2,0):"COW"},{(1,0):"WHEAT"}),
                         [["PICKUP","COW",1]])

    def test_place_precedes_ordinary_planting(self):
        obs=observation()
        obs["private"]["inventories"]=[{"COW":1}]
        obs["farms"][0]["tiles"][0][0]={"kind":"PASTURE"}
        self.assertEqual(policy().unit_actions(obs,{(0,0):"COW"},{(1,0):"WHEAT"}),
                         [["PLACE","COW"]])

    def test_stochastic_actor_cannot_defer_ready_placement_for_plant(self):
        obs=observation()
        obs["private"]["inventories"]=[{"COW":1}]
        obs["farms"][0]["tiles"][0][0]={"kind":"PASTURE"}
        p=policy()
        p.deterministic=False
        for _ in range(12):
            self.assertEqual(p.unit_actions(obs,{(0,0):"COW"},{(1,0):"WHEAT"}),
                             [["PLACE","COW"]])

    def test_no_stock_never_emits_invalid_animal_pickup_or_placement(self):
        obs=observation()
        obs["private"]["shed"]={}
        obs["private"]["seeds"]={}
        obs["farms"][0]["tiles"][0][0]={"kind":"PASTURE"}
        self.assertEqual(policy().unit_actions(obs,{(0,0):"COW"},{}),[["PASS"]])

    def test_remote_placement_keeps_its_route(self):
        obs=observation()
        obs["private"]["inventories"]=[{"COW":1}]
        obs["farms"][0]["tiles"][0][2]={"kind":"PASTURE"}
        p=policy()
        for x,expected in ((0,["EAST"]),(1,["EAST"]),(2,["PLACE","COW"])):
            obs["farms"][0]["farmer"]=[x,0]
            self.assertEqual(p.unit_actions(obs,{(2,0):"COW"},{(3,0):"WHEAT"}),[expected])

    def test_planned_weed_is_cleared_before_ordinary_planting(self):
        obs=observation()
        obs["farms"][0]["tiles"][0][0]={"kind":"WEED"}
        self.assertEqual(policy().unit_actions(obs,{(0,0):"COW"},{(1,0):"WHEAT"}),[["DIG"]])

    def test_critical_feed_still_precedes_animal_setup(self):
        obs=_obs(worker_x=3,fed=False,wheat=1)
        obs["private"]["shed"]={"WHEAT":20,"COW":1}
        obs["farms"][0]["tiles"][0][3]["consecutive_unfed"]=1
        self.assertEqual(policy().unit_actions(obs,{(0,0):"COW"},{}),[["FEED"]])

    def test_critical_water_still_precedes_animal_setup(self):
        obs=observation()
        obs["farms"][0]["tiles"][0][0]={
            "kind":"PLANT","crop":"WHEAT","planted_day":0,"yield_units":0,
            "watered_today":False,"consecutive_unwatered":1,"fertilized_until_day":0}
        self.assertEqual(policy().unit_actions(obs,{(1,0):"COW"},{}),[["WATER"]])

    def test_first_loss_harvest_still_precedes_animal_setup(self):
        obs=observation(hour=22)
        obs["day"]=4
        obs["farms"][0]["tiles"][0][0]={
            "kind":"PLANT","crop":"WHEAT","planted_day":0,"yield_units":4,
            "watered_today":True,"consecutive_unwatered":0,"fertilized_until_day":4,
            "max_lifespan_step":119}
        self.assertEqual(policy().unit_actions(obs,{(1,0):"COW"},{}),[["HARVEST"]])

    def test_setup_does_not_steal_last_reachable_normal_feed_turn(self):
        obs=_obs(worker_x=3,fed=False,wheat=1)
        obs["hour"]=23
        obs["farms"][0]["hands"]=[[0,0]]
        obs["private"]["inventories"].append({})
        p=policy()
        actions=p.unit_actions(obs,{(3,0):"COW",(0,0):"COW"},{})
        self.assertEqual(actions[0],["FEED"])
        self.assertEqual(actions[1],["BUILD_PASTURE"])

    def test_placement_rejected_when_next_day_maintenance_is_overloaded(self):
        obs=observation()
        animal=_obs()["farms"][0]["tiles"][0][3]
        obs["farms"][0]["tiles"]=[[{"kind":"PASTURE"}]+[copy.deepcopy(animal) for _ in range(16)]]
        obs["private"]["inventories"]=[{"COW":1}]
        self.assertFalse(policy().feasible(obs,0,Task((0,0),"PLACE_ANIMAL","COW",planned=True),
                                          obs["private"]["seeds"],obs["private"]["shed"]))

    def test_placement_on_last_hour_can_feed_tomorrow(self):
        obs=observation(hour=23)
        obs["farms"][0]["tiles"][0][0]={"kind":"PASTURE"}
        obs["private"]["inventories"]=[{"COW":1}]
        self.assertEqual(policy().unit_actions(obs,{(0,0):"COW"},{}),[["PLACE","COW"]])

    def test_pass_is_masked_even_for_a_pass_loving_actor(self):
        obs=_obs(worker_x=3,fed=False,wheat=1)
        p=policy(_PassLovingModel(),collect=True)
        self.assertEqual(p.unit_actions(obs,{},{}),[["FEED"]])
        self.assertEqual(p.turn_avoidable_pass,[False])
        pass_col=CANDIDATE_FEATURE_NAMES.index("op_PASS")
        self.assertFalse(p.pending.subdecisions[0].candidates[:,pass_col].any())

    def test_necessary_pass_remains_for_unreachable_worker(self):
        obs=observation(hour=23)
        obs["private"]["shed"]={}
        obs["private"]["seeds"]={}
        obs["farms"][0]["hands"]=[[3,0]]
        obs["private"]["inventories"].append({})
        self.assertEqual(policy(_PassLovingModel()).unit_actions(obs,{(0,0):"COW"},{}),
                         [["BUILD_PASTURE"],["PASS"]])


if __name__=="__main__":
    unittest.main()
