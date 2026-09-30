"""Confirmed useful planting receives an isolated, scale-independent signal."""
import copy
import unittest

import numpy as np
import torch

from test_worker_route_commitment import _Executor, _PlantLovingModel, _obs
from train_v25_worker_ppo import actor_samples_and_advantages
from worker_policy import (
    PLANNED_PLANT_COMPLETION_ACTOR_BONUS, PLANNED_PLANT_REWARD_EQUIV,
    SubDecision, TurnRecord, WorkerPolicy,
)
from worker_reward import RewardBreakdown, compute_worker_reward


class PlantSignalTest(unittest.TestCase):
    def setup_plant(self, day=0, target=(0,0)):
        obs=_obs(fed=True)
        obs["day"]=day
        obs["private"]["seeds"]={"WHEAT":1}
        policy=WorkerPolicy(_Executor(),_PlantLovingModel(),torch.device("cpu"),True,True)
        actions=policy.unit_actions(obs,{}, {target:"WHEAT"})
        return obs,policy,actions,policy.pending.subdecisions[0]

    def test_engine_confirmed_completion_credits_only_origin_and_leaves_turn_reward(self):
        obs,p,actions,origin=self.setup_plant()
        after=copy.deepcopy(obs)
        after["farms"][0]["tiles"][0][0]={
            "kind":"PLANT","crop":"WHEAT","planted_day":0,"yield_units":1}
        after["private"]["seeds"]["WHEAT"]=0
        reward=compute_worker_reward(_Executor(),obs,
            {"farmer":actions[0],"hands":[],"_plan_credit":p.turn_plan_credit},after)
        self.assertEqual(reward.planned_plants_completed,1)
        self.assertEqual(origin.actor_bonus,0)
        unrelated=SubDecision(origin.candidates.copy(),0,0)
        p.pending.subdecisions.append(unrelated)
        p.finish_turn(reward)
        self.assertEqual(origin.actor_bonus,PLANNED_PLANT_COMPLETION_ACTOR_BONUS)
        self.assertEqual(origin.plant_completion_actor_bonus,PLANNED_PLANT_COMPLETION_ACTOR_BONUS)
        self.assertEqual(origin.reward_equiv_bonus,PLANNED_PLANT_REWARD_EQUIV)
        self.assertEqual(unrelated.actor_bonus,0)
        self.assertEqual(p.records[-1].reward,reward.reward)
        self.assertEqual(p.records[-1].reward,1)
        with self.assertRaises(RuntimeError):
            p.finish_turn(reward)
        self.assertEqual(origin.actor_bonus,PLANNED_PLANT_COMPLETION_ACTOR_BONUS)

    def test_failed_plant_gets_no_actor_bonus(self):
        obs,p,actions,origin=self.setup_plant()
        reward=compute_worker_reward(_Executor(),obs,
            {"farmer":actions[0],"hands":[],"_plan_credit":p.turn_plan_credit},obs)
        p.finish_turn(reward)
        self.assertEqual(origin.actor_bonus,0)
        self.assertEqual(origin.plant_completion_actor_bonus,0)
        self.assertEqual(origin.reward_equiv_bonus,0)

    def test_late_plant_gets_no_actor_bonus(self):
        obs,p,actions,origin=self.setup_plant(day=28)
        after=copy.deepcopy(obs)
        after["farms"][0]["tiles"][0][0]={
            "kind":"PLANT","crop":"WHEAT","planted_day":28,"yield_units":1}
        after["private"]["seeds"]["WHEAT"]=0
        reward=compute_worker_reward(_Executor(),obs,
            {"farmer":actions[0],"hands":[],"_plan_credit":p.turn_plan_credit},after)
        self.assertEqual(reward.unproductive_plants_created,1)
        p.finish_turn(reward)
        self.assertEqual(origin.actor_bonus,0)
        self.assertEqual(origin.reward_equiv_bonus,0)

    def test_remote_route_credits_original_choice_after_completion(self):
        obs,p,actions,origin=self.setup_plant(target=(2,0))
        self.assertEqual(actions,[["EAST"]])
        p.finish_turn(RewardBreakdown())
        self.assertEqual(origin.actor_bonus,0)
        at_target=copy.deepcopy(obs)
        at_target["farms"][0]["farmer"]=[2,0]
        at_target["hour"]+=2
        self.assertEqual(p.unit_actions(at_target,{}, {(2,0):"WHEAT"}),[["PLANT","WHEAT"]])
        self.assertEqual(p.pending.subdecisions,[])
        p.finish_turn(RewardBreakdown(reward=1,planned_plants_completed=1,
                                    planned_plants_completed_by_worker={"0":1}))
        self.assertIs(p.records[0].subdecisions[0],origin)
        self.assertEqual(origin.actor_bonus,PLANNED_PLANT_COMPLETION_ACTOR_BONUS)
        self.assertEqual(p.records[0].reward,0)
        self.assertEqual(p.records[1].reward,1)

    def test_interrupted_route_gets_no_completion_bonus(self):
        obs,p,actions,origin=self.setup_plant(target=(2,0))
        p.finish_turn(RewardBreakdown())
        no_seed=copy.deepcopy(obs)
        no_seed["hour"]+=1
        no_seed["private"]["seeds"]={}
        p.unit_actions(no_seed,{}, {})
        p.finish_turn(RewardBreakdown())
        self.assertEqual(origin.actor_bonus,0)
        self.assertEqual(origin.plant_completion_actor_bonus,0)

    def test_mixed_late_and_useful_completions_credit_the_correct_worker(self):
        obs=_obs(fed=True)
        obs["day"]=22
        obs["farms"][0]["hands"]=[[1,0]]
        obs["private"]["inventories"].append({})
        obs["private"]["seeds"]={"MELON":1,"WHEAT":1}
        e=_Executor()
        e.CROPS={**e.CROPS,"MELON":(80,250,((10,6),),10)}
        e.CROP_FIRST_YIELD_DAY={**e.CROP_FIRST_YIELD_DAY,"MELON":10}
        p=WorkerPolicy(e,_PlantLovingModel(),torch.device("cpu"),True,True)
        actions=p.unit_actions(obs,{}, {(0,0):"MELON",(1,0):"WHEAT"})
        self.assertEqual(actions,[["PLANT","MELON"],["PLANT","WHEAT"]])
        origins={w:origin for w,origin in p.turn_planned_plant_origins}
        after=copy.deepcopy(obs)
        for x,crop in ((0,"MELON"),(1,"WHEAT")):
            after["farms"][0]["tiles"][0][x]={
                "kind":"PLANT","crop":crop,"planted_day":22,"yield_units":1}
        after["private"]["seeds"]={}
        reward=compute_worker_reward(e,obs,{"farmer":actions[0],"hands":actions[1:],
                    "_plan_credit":p.turn_plan_credit},after)
        self.assertEqual(reward.planned_plants_completed,1)
        self.assertEqual(reward.planned_plants_completed_by_worker,{"1":1})
        self.assertEqual(reward.unproductive_plants_created,1)
        p.finish_turn(reward)
        self.assertEqual(origins[0].actor_bonus,0)
        self.assertEqual(origins[0].reward_equiv_bonus,0)
        self.assertEqual(origins[1].actor_bonus,PLANNED_PLANT_COMPLETION_ACTOR_BONUS)
        self.assertEqual(origins[1].reward_equiv_bonus,PLANNED_PLANT_REWARD_EQUIV)

    def test_fixed_component_survives_large_gae_scales_without_double_counting(self):
        for scale in (400,1000,4000):
            plant=SubDecision(np.zeros((2,6),np.float16),0,0,
                actor_bonus=PLANNED_PLANT_COMPLETION_ACTOR_BONUS,
                reward_equiv_bonus=PLANNED_PLANT_REWARD_EQUIV,
                plant_completion_actor_bonus=PLANNED_PLANT_COMPLETION_ACTOR_BONUS)
            other=SubDecision(np.zeros((2,6),np.float16),0,0)
            records=[TurnRecord(np.zeros(4),[plant],0,0),TurnRecord(np.zeros(4),[other],0,1)]
            records[0].advantage=-scale
            records[1].advantage=scale
            _,_,_,bonuses,actual_scale,_,_=actor_samples_and_advantages(records)
            self.assertEqual(actual_scale,scale)
            self.assertAlmostEqual(float(bonuses[0]),0.25+16/scale,places=6)
            self.assertEqual(float(bonuses[1]),0)


if __name__=="__main__":
    unittest.main()
