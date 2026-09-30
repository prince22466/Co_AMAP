"""Regression coverage for first-loss deadlines, capacity and decay accounting."""
from __future__ import annotations

import copy
import unittest

import torch

from test_worker_route_commitment import _Executor, _PassLovingModel, _decay_crop_obs, _obs
from test_worker_reward_contract import EXECUTOR, _obs as reward_obs
from worker_policy import CANDIDATE_FEATURE_NAMES, Task, WorkerPolicy, _crop_decay_deadline_step
from worker_reward import (
    CROP_TO_WEED_PENALTY, LOST_HARVESTABLE_UNIT_PENALTY,
    RewardBreakdown, compute_worker_reward,
)


class _OngoingExecutor(_Executor):
    CROPS = {**_Executor.CROPS, "TOMATO": (50,60,((8,2),(9,2),(10,2),(11,2)),11)}
    CROP_FIRST_YIELD_DAY = {**_Executor.CROP_FIRST_YIELD_DAY, "TOMATO": 8}


class _WrongPairLovingModel(_PassLovingModel):
    def logits(self, candidates):
        # Prefer giving worker 0 the crop at x=1, although that assignment
        # strands the x=0 crop that only this worker can reach before expiry.
        target = CANDIDATE_FEATURE_NAMES.index("target_x")
        worker = CANDIDATE_FEATURE_NAMES.index("worker_x")
        return candidates[:,target]*1000-candidates[:,worker]*100


def policy(model=None, executor=None):
    return WorkerPolicy(executor or _Executor(),model or _PassLovingModel(),
                        torch.device("cpu"),deterministic=True,collect=False)


def crop(units=4, start=120, name="WHEAT", planted=0):
    return {"kind":"PLANT","crop":name,"planted_day":planted,
            "yield_units":units,"watered_today":True,"consecutive_unwatered":0,
            "max_lifespan_step":start,"fertilized_until_day":-1}


class LosslessSchedulerTest(unittest.TestCase):
    def test_four_units_do_not_extend_first_loss_deadline(self):
        obs=_decay_crop_obs(worker_x=1,hour=22,max_lifespan_step=120)
        obs["farms"][0]["tiles"][0][0]["yield_units"]=4
        self.assertEqual(_crop_decay_deadline_step(_Executor(),obs,obs["farms"][0]["tiles"][0][0]),120)
        p=policy()
        self.assertEqual(p.unit_actions(obs,{},{}),[["WEST"]])
        self.assertEqual(p.active_tasks[0].op,"HARVEST")

    def test_live_crop_outside_current_plan_still_gets_critical_water(self):
        obs=_obs(worker_x=0,fed=True,wheat=0)
        obs["day"]=1
        obs["farms"][0]["tiles"][0][0]=crop(units=0)
        obs["farms"][0]["tiles"][0][0].update(watered_today=False,consecutive_unwatered=1)
        self.assertEqual(policy().unit_actions(obs,{},{}),[["WATER"]])

    def test_queue_pressure_starts_before_individual_last_departure(self):
        obs=_obs(worker_x=2,fed=True,wheat=0)
        obs.update(day=4,hour=20)
        obs["farms"][0]["tiles"][0][0]=crop()
        obs["farms"][0]["tiles"][0][1]=crop()
        p=policy()
        # At step 116 neither crop is individually urgent. Waiting nevertheless
        # prevents one worker completing travel+HARVEST for both before rollover.
        self.assertEqual(p.unit_actions(obs,{},{}),[["WEST"]])
        self.assertEqual(p.active_tasks[0].target,(1,0))

    def test_urgent_assignment_preserves_specialist_worker(self):
        obs=_obs(worker_x=0,fed=True,wheat=0)
        obs.update(day=4,hour=22)
        obs["farms"][0]["hands"]=[[2,0]]
        obs["private"]["inventories"]=[{},{}]
        obs["farms"][0]["tiles"][0][0]=crop(start=119)
        obs["farms"][0]["tiles"][0][1]=crop(start=119)
        actions=policy(_WrongPairLovingModel()).unit_actions(obs,{},{})
        self.assertEqual(actions,[["HARVEST"],["WEST"]])

    def test_missing_first_deadline_does_not_disallow_salvage(self):
        obs=_decay_crop_obs(worker_x=1,hour=2,max_lifespan_step=96)
        obs["farms"][0]["tiles"][0][0]["yield_units"]=3
        self.assertEqual(policy().unit_actions(obs,{},{}),[["WEST"]])

    def test_plant_rejected_when_existing_feed_and_water_fill_day(self):
        obs=_obs(worker_x=0,fed=False)
        obs["hour"]=22
        obs["private"]["seeds"]={"WHEAT":1}
        self.assertFalse(policy().feasible(obs,0,Task((0,0),"PLANT","WHEAT",planned=True),
                                          obs["private"]["seeds"],obs["private"]["shed"]))

    def test_feed_wheat_renewal_is_not_blocked_by_future_stock_shortage(self):
        obs=_obs(worker_x=0,fed=True,wheat=0)
        obs["private"]["shed"]={"WHEAT":0}
        obs["private"]["seeds"]={"WHEAT":1}
        self.assertTrue(policy().feasible(obs,0,Task((0,0),"PLANT","WHEAT",planned=True),
                                         obs["private"]["seeds"],obs["private"]["shed"]))

    def test_empty_shed_critical_feed_can_source_mature_field_wheat(self):
        obs=_obs(worker_x=0,fed=False,wheat=0)
        obs.update(day=4,hour=10)
        obs["private"]["shed"]={"WHEAT":0}
        obs["farms"][0]["tiles"][0][0]=crop(start=144)
        obs["farms"][0]["tiles"][0][3]["consecutive_unfed"]=1
        self.assertEqual(policy().unit_actions(obs,{},{}),[["HARVEST"]])

    def test_emergency_field_harvest_reserves_a_followup_feed_turn(self):
        obs=_obs(worker_x=0,fed=False,wheat=0)
        obs.update(day=4,hour=21)
        obs["private"]["shed"]={"WHEAT":0}
        obs["farms"][0]["tiles"][0][0]=crop(start=144)
        obs["farms"][0]["tiles"][0][3]["consecutive_unfed"]=1
        task=next(t for t in policy().tasks(obs,{}, {}) if t.op=="HARVEST")
        self.assertFalse(policy().feasible(obs,0,task,{},{}))

    def test_plant_admitted_when_followup_water_and_maintenance_fit(self):
        obs=_obs(worker_x=0,fed=True)
        obs["hour"]=22
        obs["private"]["seeds"]={"WHEAT":1}
        self.assertTrue(policy().feasible(obs,0,Task((0,0),"PLANT","WHEAT",planned=True),
                                         obs["private"]["seeds"],obs["private"]["shed"]))

    def test_plant_rejected_by_next_day_maintenance_overload(self):
        obs=_obs(worker_x=0,fed=True)
        animal=copy.deepcopy(obs["farms"][0]["tiles"][0][3])
        obs["farms"][0]["tiles"]=[[None]+[copy.deepcopy(animal) for _ in range(16)]]
        obs["private"]["shed"]={"WHEAT":100}
        obs["private"]["seeds"]={"WHEAT":1}
        self.assertFalse(policy().feasible(obs,0,Task((0,0),"PLANT","WHEAT",planned=True),
                                          obs["private"]["seeds"],obs["private"]["shed"]))

    def test_exhausted_ongoing_crop_is_retired_before_zero_yield_decay(self):
        obs=_obs(worker_x=0,fed=True,wheat=0)
        obs.update(day=11,hour=23)
        obs["farms"][0]["tiles"][0][0]=crop(units=0,start=288,name="TOMATO")
        self.assertEqual(policy(executor=_OngoingExecutor()).unit_actions(obs,{},{}),[["DIG"]])

    def test_final_ongoing_harvest_leaves_a_cleanup_turn(self):
        obs=_obs(worker_x=0,fed=True,wheat=0)
        obs.update(day=11,hour=22)
        tile=crop(start=288,name="TOMATO")
        obs["farms"][0]["tiles"][0][0]=tile
        self.assertEqual(_crop_decay_deadline_step(_OngoingExecutor(),obs,tile),286)

    def test_final_yield_is_harvested_promptly_and_cleanup_is_committed(self):
        obs=_obs(worker_x=0,fed=True,wheat=0)
        obs.update(day=11,hour=0)
        obs["farms"][0]["tiles"][0][0]=crop(start=288,name="TOMATO")
        p=policy(executor=_OngoingExecutor())
        self.assertEqual(p.unit_actions(obs,{},{}),[["HARVEST"]])
        self.assertEqual(p.active_tasks[0].op,"DIG")
        obs["hour"]=1
        obs["farms"][0]["tiles"][0][0]["yield_units"]=0
        self.assertEqual(p.unit_actions(obs,{},{}),[["DIG"]])
        self.assertNotIn(0,p.active_tasks)

    def test_ongoing_crop_is_not_retired_before_its_final_production_day(self):
        obs=_obs(worker_x=0,fed=True,wheat=0)
        obs.update(day=10,hour=23)
        obs["farms"][0]["tiles"][0][0]=crop(start=288,name="TOMATO")
        p=policy(executor=_OngoingExecutor())
        self.assertIsNone(p._retirement_after_harvest(obs,Task((0,0),"HARVEST","TOMATO")))


class SpoilageAccountingTest(unittest.TestCase):
    def transition(self,units_after,step,units_before=4,harvest=False):
        before=reward_obs(crop(units_before),day=step//24,hour=step%24)
        tile={"kind":"WEED"} if units_after is None else crop(units_after)
        after=reward_obs(tile,day=(step+1)//24,hour=(step+1)%24)
        return compute_worker_reward(EXECUTOR,before,{"farmer":["HARVEST" if harvest else "PASS"],"hands":[]},after)

    def test_surviving_crop_decay_is_counted_and_penalized(self):
        r=self.transition(3,120)
        self.assertEqual(r.lost_harvestable_units,1)
        self.assertEqual(r.crop_units_lost_to_decay,1)
        self.assertEqual(r.crop_units_lost_to_decay_by_crop["WHEAT"],1)
        self.assertEqual(r.reward,LOST_HARVESTABLE_UNIT_PENALTY)

    def test_complete_decay_counts_each_unit_once(self):
        total=RewardBreakdown()
        held=4
        for step in range(120,127):
            after=held-1 if step%2==0 else held
            total.add(self.transition(after if after else None,step,held))
            held=after
        self.assertEqual(total.lost_harvestable_units,4)
        self.assertEqual(total.crop_units_lost_to_decay,4)
        self.assertEqual(total.crops_to_weed,1)
        self.assertEqual(total.reward,CROP_TO_WEED_PENALTY+4*LOST_HARVESTABLE_UNIT_PENALTY)

    def test_no_decay_tick_has_no_spoilage(self):
        self.assertEqual(self.transition(3,121,3).lost_harvestable_units,0)

    def test_successful_harvest_before_disappearance_is_not_lost_yield(self):
        before=reward_obs(crop(),day=5,hour=0)
        after=reward_obs(None,inventory={"WHEAT":4},day=5,hour=1)
        r=compute_worker_reward(EXECUTOR,before,{"farmer":["HARVEST"],"hands":[]},after)
        self.assertEqual(r.lost_harvestable_units,0)
        self.assertEqual(r.crop_units_harvested_total,4)

    def test_ongoing_harvest_at_expiry_does_not_double_count_collected_units(self):
        before=reward_obs(crop(name="TOMATO",start=288),day=12,hour=0)
        after=reward_obs({"kind":"WEED"},inventory={"TOMATO":4},day=12,hour=1)
        r=compute_worker_reward(_OngoingExecutor(),before,{"farmer":["HARVEST"],"hands":[]},after)
        self.assertEqual(r.crops_to_weed,1)
        self.assertEqual(r.lost_harvestable_units,0)
        self.assertEqual(r.crop_units_harvested_total,4)

    def test_final_production_day_empty_crop_cleanup_is_not_crop_death(self):
        before=reward_obs(crop(0,name="TOMATO",start=288),day=11,hour=12)
        after=reward_obs(None,day=11,hour=13)
        r=compute_worker_reward(_OngoingExecutor(),before,{"farmer":["DIG"],"hands":[]},after)
        self.assertEqual(r.crops_died,0)
        self.assertEqual(r.lost_harvestable_units,0)

    def test_random_weed_after_exhausted_cleanup_is_not_crop_loss(self):
        before=reward_obs(crop(0,name="TOMATO",start=288),day=11,hour=23)
        after=reward_obs({"kind":"WEED"},day=12,hour=0)
        r=compute_worker_reward(_OngoingExecutor(),before,{"farmer":["DIG"],"hands":[]},after)
        self.assertEqual(r.crops_to_weed,0)
        self.assertEqual(r.crops_died,0)

    def test_random_weed_does_not_hide_productive_crop_destruction(self):
        before=reward_obs(crop(),day=4,hour=23)
        after=reward_obs({"kind":"WEED"},day=5,hour=0)
        r=compute_worker_reward(EXECUTOR,before,{"farmer":["DIG"],"hands":[]},after)
        self.assertEqual(r.crops_to_weed,0)
        self.assertEqual(r.crops_died,1)
        self.assertEqual(r.lost_harvestable_units,4)

    def test_melon_fallback_uses_engine_lifespan_not_planner_max_yield_age(self):
        tile=crop(name="MELON",start=-1)
        self.assertEqual(_crop_decay_deadline_step(_Executor(),{"day":10,"hour":0},tile),13*24)


if __name__=="__main__":
    unittest.main()
