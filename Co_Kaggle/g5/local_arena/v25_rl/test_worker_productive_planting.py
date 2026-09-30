"""Useful planting, legitimate deferral and tile-time measurement regressions."""
import unittest

import torch

from test_worker_route_commitment import _Executor, _PlantLovingModel, _obs
from test_worker_reward_contract import EXECUTOR, _obs as reward_obs
from worker_policy import CANDIDATE_FEATURE_NAMES, Task, WorkerPolicy
from worker_reward import compute_worker_reward, measure_land_use, plant_can_deliver_before_end
from train_v25_worker_ppo import EpisodeResult, summary, compose_environment_action


class ProductivePlantingTest(unittest.TestCase):
    def policy(self, collect=False):
        return WorkerPolicy(_Executor(), _PlantLovingModel(), torch.device("cpu"),
                            deterministic=True, collect=collect)

    def test_last_useful_wheat_day_and_later_planting(self):
        obs = _obs()
        obs.update(day=27, hour=0)
        self.assertTrue(plant_can_deliver_before_end(_Executor(), obs, "WHEAT", (3,0)))
        obs["day"] = 28
        self.assertFalse(plant_can_deliver_before_end(_Executor(), obs, "WHEAT", (0,0)))

    def test_remote_crop_needs_time_to_return_and_deliver(self):
        obs = _obs()
        obs.update(day=27, hour=0)
        self.assertTrue(plant_can_deliver_before_end(_Executor(), obs, "WHEAT", (11,0)))
        self.assertFalse(plant_can_deliver_before_end(_Executor(), obs, "WHEAT", (12,0)))

    def test_engine_allows_first_wheat_harvest_on_last_game_day(self):
        from kaggle_environments.envs.kaggriculture import kaggriculture as engine
        obs=reward_obs(None,day=27)
        farm=obs["farms"][0]
        farm["tiles"]=[[None for _ in range(5)] for _ in range(5)]
        obs["private"]["seeds"]={"WHEAT":1}
        engine._apply_unit_action(farm,obs["private"],0,["PLANT","WHEAT"],5,27,24,100)
        self.assertEqual(farm["tiles"][0][0]["planted_day"],27)
        engine._apply_unit_action(farm,obs["private"],0,["HARVEST"],5,29,24,100)
        self.assertIsNone(farm["tiles"][0][0])
        self.assertGreater(obs["private"]["inventories"][0].get("WHEAT",0),0)

    def test_late_seed_does_not_make_optional_work_an_avoidable_delay(self):
        obs = _obs(fed=True)
        obs.update(day=28, hour=0)
        obs["private"]["seeds"] = {"WHEAT":1}
        p = self.policy()
        self.assertTrue(p.feasible(obs,0,Task((0,0),"PLANT","WHEAT",planned=True),
                                   obs["private"]["seeds"],obs["private"]["shed"]))
        actions = p.unit_actions(obs,{}, {(0,0):"WHEAT"})
        self.assertEqual(actions[0][0],"PLANT")
        self.assertEqual(p.turn_avoidable_plant_delay,[False])
        other=Task((0,0),"COLLECT_FERTILIZER","FERTILIZER")
        plant=Task((0,0),"PLANT","WHEAT",planned=True)
        self.assertFalse(p.avoidable_plant_delay(0,other,[(0,plant,plant.key)]))

    def test_unproductive_plant_route_has_no_movement_credit(self):
        obs=_obs(fed=True)
        obs.update(day=28,hour=0)
        obs["private"]["seeds"]={"WHEAT":1}
        p=self.policy()
        self.assertEqual(p.unit_actions(obs,{}, {(1,0):"WHEAT"}),[["EAST"]])
        self.assertEqual(p.turn_route_progress,[False])

    def test_long_growth_crops_use_engine_first_yield_age(self):
        e = _Executor()
        e.CROP_FIRST_YIELD_DAY = {**e.CROP_FIRST_YIELD_DAY,"TOMATO":8,"STRAWBERRY":10}
        obs = _obs()
        for crop,last_day in (("TOMATO",21),("STRAWBERRY",19)):
            obs["day"] = last_day
            self.assertTrue(plant_can_deliver_before_end(e,obs,crop,(0,0)))
            obs["day"] += 1
            self.assertFalse(plant_can_deliver_before_end(e,obs,crop,(0,0)))

    def test_no_plant_completion_credit_for_immature_end_game_crop(self):
        before = reward_obs(None,day=28)
        before["private"]["seeds"] = {"WHEAT":1}
        after = reward_obs({"kind":"PLANT","crop":"WHEAT","planted_day":28,
                            "yield_units":0},day=28)
        r = compute_worker_reward(EXECUTOR,before,{"farmer":["PLANT","WHEAT"],"hands":[],
                    "_plan_credit":[{"op":"PLANT","item":"WHEAT"}]},after)
        self.assertEqual(r.seeds_planted_total,1)
        self.assertEqual(r.unproductive_plants_created,1)
        self.assertEqual(r.planned_plants_completed,0)
        self.assertEqual(r.reward,0)

    def test_maintenance_and_output_do_not_receive_delay_penalty(self):
        p = self.policy()
        plant = Task((1,0),"PLANT","WHEAT",planned=True)
        for op in ("WATER","FEED","CARE","HARVEST","DELIVER","PLACE_ANIMAL"):
            other = Task((0,0),op,"COW" if op=="PLACE_ANIMAL" else "WHEAT",planned=True)
            choices = [(0,plant,plant.key),(0,other,other.key)]
            self.assertFalse(p.avoidable_plant_delay(0,other,choices),op)
            self.assertEqual(p.actor_bonus_for_choice(0,other,choices),0,op)

    def test_optional_work_can_receive_delay_penalty_only_with_plant_alternative(self):
        p = self.policy()
        plant = Task((1,0),"PLANT","WHEAT",planned=True)
        other = Task((0,0),"COLLECT_FERTILIZER","FERTILIZER")
        choices = [(0,plant,plant.key),(0,other,other.key)]
        self.assertTrue(p.avoidable_plant_delay(0,other,choices))
        self.assertLess(p.actor_bonus_for_choice(0,other,choices),0)
        self.assertFalse(p.avoidable_plant_delay(1,other,choices))
        self.assertEqual(p.actor_bonus_for_choice(0,other,choices[1:]),0)

    def test_actual_optional_assignment_records_delay(self):
        class FertLovingModel(_PlantLovingModel):
            def logits(self,candidates):
                return candidates[:,CANDIDATE_FEATURE_NAMES.index("op_FERTILIZE")] * 100
        obs = _obs(fed=True)
        obs["private"]["inventories"][0]["FERTILIZER"] = 1
        obs["private"]["seeds"] = {"WHEAT":1}
        obs["farms"][0]["tiles"][0][0] = {
            "kind":"PLANT","crop":"WHEAT","planted_day":0,"yield_units":0,
            "watered_today":True,"fertilized_until_day":-1}
        p = WorkerPolicy(_Executor(),FertLovingModel(),torch.device("cpu"),True,True)
        self.assertEqual(p.unit_actions(obs,{}, {(1,0):"WHEAT"}),[["FERTILIZE"]])
        self.assertEqual(p.turn_avoidable_plant_delay,[True])
        self.assertLess(p.pending.subdecisions[0].actor_bonus,0)

    def test_land_counts_exclude_locked_and_distinguish_seed_backed_empty(self):
        obs = _obs()
        obs.update(day=5,hour=0)
        obs["farms"][0]["tiles"] = [[None,None,None,"LOCKED",
            {"kind":"PLANT","crop":"WHEAT","planted_day":5,"yield_units":0},
            {"kind":"PLANT","crop":"WHEAT","planted_day":0,"yield_units":0},
            {"kind":"WEED"}]]
        e = _Executor()
        e.ANIMAL_POINTS = {(2,0)}
        counts = measure_land_use(e,obs,{(0,0):"WHEAT"})
        self.assertEqual(counts["owned_tile_turns"],6)
        self.assertEqual(counts["empty_tile_turns"],3)
        self.assertEqual(counts["empty_crop_tile_turns"],2)
        self.assertEqual(counts["empty_animal_reserved_tile_turns"],1)
        self.assertEqual(counts["productive_crop_tile_turns"],1)
        self.assertEqual(counts["seed_backed_empty_crop_tile_turns"],1)
        self.assertEqual(counts["unseeded_empty_crop_tile_turns"],1)
        self.assertEqual(counts["midgame_owned_tile_turns"],6)
        obs["day"] = 25
        self.assertEqual(measure_land_use(e,obs,{})["midgame_owned_tile_turns"],0)

    def test_summary_weights_by_tile_time_and_metadata_never_reaches_engine(self):
        def result(name,empty,total):
            return EpisodeResult(name,None,True,0,0,1,
                {"empty_tile_turns":empty,"owned_tile_turns":total},0,0)
        s = summary([result("small",1,2),result("large",0,8)],"test")
        self.assertEqual(s["owned_empty_tile_fraction"],0.1)
        self.assertIsNone(s["crop_empty_tile_fraction"])
        action = {"farmer":["PASS"],"hands":[],"_land_use":{"owned_tile_turns":3},
                  "_avoidable_plant_delay":[True]}
        self.assertEqual(compose_environment_action(action,[]),
                         {"farmer":["PASS"],"hands":[],"market":[]})


if __name__ == "__main__":
    unittest.main()
