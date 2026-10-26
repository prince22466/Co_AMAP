"""Promotion policy tests; no inference or benchmark execution."""
import unittest
from run_submission20_pilot import baseline_ready,readiness,PILOT_IDS

class PilotPolicy(unittest.TestCase):
    def test_behavioral_baseline_failure_is_expected(self):
        self.assertTrue(baseline_ready(dict(test_exit_code=1,test_output='2 failed, 3 passed',error=None)))
    def test_collection_and_fixture_failures_block_before_gpu(self):
        for output in ["ModuleNotFoundError: No module named 'inline_snapshot'",
                       "recursive dependency involving fixture 'httpbin' detected",'ERROR at setup of test_x']:
            self.assertFalse(baseline_ready(dict(test_exit_code=1,test_output=output)))
    def test_empty_missing_or_zero_collected_tests_block(self):
        for row in [{},dict(test_exit_code=0,test_output=''),dict(test_exit_code=5,test_output='no tests ran')]:
            self.assertFalse(baseline_ready(row))
    def test_scripted_smoke_cannot_unlock_larger_run(self):
        rows=[dict(instance_id=i,resolved=True,agent_patch='diff') for i in PILOT_IDS]
        self.assertFalse(readiness(dict(passed=True,scope='scripted mechanics; no inference'),rows))
        self.assertTrue(readiness(dict(passed=True,scope='real E4B synthetic smoke; not benchmark'),rows))
    def test_official_error_message_field_blocks_promotion(self):
        self.assertFalse(baseline_ready(dict(test_exit_code=0,test_output='1 passed',error_message='execution failure')))
        smoke=dict(passed=True,scope='real E4B synthetic smoke; not benchmark')
        rows=[dict(task_id=i,resolved=True,agent_patch='diff') for i in PILOT_IDS]
        rows[0]['error_message']='budget exhausted'
        self.assertFalse(readiness(smoke,rows))
    def test_no_cherry_picking_or_patches_as_success(self):
        smoke=dict(passed=True,scope='real E4B synthetic smoke; not benchmark')
        rows=[dict(instance_id=i,resolved=True,agent_patch='diff') for i in PILOT_IDS]
        self.assertFalse(readiness(smoke,rows[:-1]))
        rows[0]['resolved']=False;self.assertFalse(readiness(smoke,rows))

if __name__=='__main__':unittest.main()
