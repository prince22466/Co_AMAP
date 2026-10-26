"""Pilot policy tests; no model inference."""
import unittest
from run_submission21_pilot import baseline_ready,pilot_metrics,readiness,PILOT_IDS

def row(task,solved=False,patch='diff',trace=None):
    return {'instance_id':task,'resolved':solved,'agent_patch':patch,'trace':trace or [{'tool':'edit_file','status':'ok'}]}

class PilotPolicy(unittest.TestCase):
    def test_expected_baseline_failure_is_allowed(self):
        self.assertTrue(baseline_ready({'test_exit_code':1,'test_output':'1 failed, 2 passed'}))
    def test_dependency_and_collection_block_gpu(self):
        for msg in ["ModuleNotFoundError: No module named 'inline_snapshot'",
                    "recursive dependency involving fixture 'httpbin'",
                    "ERROR at setup of test_x",'no tests ran']:
            self.assertFalse(baseline_ready({'test_exit_code':1,'test_output':msg}))
    def test_no_empty_or_error_baseline(self):
        self.assertFalse(baseline_ready({}))
        self.assertFalse(baseline_ready({'test_exit_code':0,'test_output':'1 passed','error_message':'oops'}))
    def test_all_five_patches_and_one_solved_required(self):
        rows=[row(t,i==0) for i,t in enumerate(PILOT_IDS)]
        self.assertTrue(readiness(rows))
        rows[1]['agent_patch']=''
        self.assertFalse(readiness(rows))
        rows[1]['agent_patch']='diff';rows[0]['resolved']=False
        self.assertFalse(readiness(rows))
    def test_missing_task_or_trace_blocks_promotion(self):
        rows=[row(t,i==0) for i,t in enumerate(PILOT_IDS)]
        self.assertFalse(readiness(rows[:-1]))
        rows[0].pop('trace')
        self.assertFalse(readiness(rows))
    def test_s20_skill_failures_block_promotion(self):
        rows=[row(t,i==0) for i,t in enumerate(PILOT_IDS)]
        rows[2]['trace']=[{'error':"Skill 'patch_applier' not found.",'error_code':'SKILL_NOT_FOUND'}]
        self.assertFalse(readiness(rows))
        self.assertGreater(pilot_metrics(rows)['tool_failure_events'],0)

if __name__=='__main__':unittest.main()
