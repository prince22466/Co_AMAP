"""Pilot policy tests (synthetic traces, not E4B benchmark results)."""
import json
import tempfile
from pathlib import Path
import unittest
from run_submission21_pilot import (
    TOOL_FAILURE,PILOT_IDS,baseline_ready,pilot_metrics,promotion_candidate,
    readiness,review_complete,review_existing,task_evidence,trace_calls)

CHECK='python -c "import pkg; assert pkg.render() == 2"'

def trace(command=CHECK,status='ok'):
    return [
        {'tool':'edit_file','arguments':{'filepath':'pkg.py'},'result':{'status':'ok'}},
        {'tool':'run_command','arguments':{'command':command},
         'result':{'status':status,'exit_code':0,'stdout':'1 passed'}},
        {'tool':'submit_patch','arguments':{},'result':{'status':'ok'}},
    ]

def row(task,solved=False,patch='diff',events=None):
    return {'instance_id':task,'resolved':solved,'agent_patch':patch,
            'trace':trace() if events is None else events}

def good_rows():
    return [row(task,index==0) for index,task in enumerate(PILOT_IDS)]

def smoke(passed=True):
    return {'passed':passed,'scope':'real-model synthetic smoke; not benchmark'}

def review():
    return {'tasks':[{'instance_id':id,'patch_relevant':True,
                       'behavior_check_relevant':True,'final_diff_reviewed':True}
                      for id in PILOT_IDS]}

class PilotPolicy(unittest.TestCase):
    def test_behavioral_baseline_failure_allowed(self):
        self.assertTrue(baseline_ready({'test_exit_code':1,'test_output':'1 failed, 2 passed'}))
    def test_collection_and_environment_failures_block(self):
        for msg in ["ModuleNotFoundError: No module named 'inline_snapshot'",
                    "recursive dependency involving fixture 'httpbin'",
                    "ERROR at setup of test_x",'no tests ran']:
            self.assertFalse(baseline_ready({'test_exit_code':1,'test_output':msg}))
        self.assertFalse(baseline_ready({}))
        self.assertFalse(baseline_ready({'test_exit_code':0,'test_output':'1 passed',
                                         'error_message':'oops'}))
    def test_s20_plain_error_messages_without_codes(self):
        for msg in ["Skill 'patch_applier' not found.",
                    "Tool 'edit' not found.",
                    "Script 'scripts/candidate' not found",
                    "Argument 'skill_name' is required",
                    "Argument 'file_path' is required",
                    'SKILL_NOT_FOUND','SCRIPT_NOT_FOUND']:
            self.assertRegex(msg,TOOL_FAILURE)
        rows=good_rows()
        rows[1]['trace'].append({'status':'error','message':"Tool 'edit' not found."})
        self.assertGreater(pilot_metrics(rows)['tool_failure_events'],0)
        self.assertFalse(promotion_candidate(rows,smoke()))
    def test_edit_only_or_weak_checks_cannot_promote(self):
        variants=[
            [{'tool':'edit_file','arguments':{},'result':{'status':'ok'}}],
            trace('python -c "assert True"'),
            trace('git diff --check'),
            trace('python -m pytest -q'),
            trace(status='error'),
            [trace()[1],trace()[0],trace()[2]],
        ]
        for events in variants:
            self.assertFalse(task_evidence(row(PILOT_IDS[0],events=events))['submit_after_check_observed'])
    def test_failed_output_with_exit_zero_is_not_success(self):
        events=trace()
        events[1]['result']['stdout']='test FAILED despite zero exit code\n'
        self.assertFalse(task_evidence(row(PILOT_IDS[0],events=events))['post_edit_check_observed'])
    def test_good_traces_require_all_checks_and_successful_smoke(self):
        rows=good_rows()
        self.assertTrue(promotion_candidate(rows,smoke()))
        self.assertFalse(promotion_candidate(rows,smoke(False)))
        rows[1]['trace']=[{'tool':'edit_file','result':{'status':'ok'}}]
        self.assertFalse(promotion_candidate(rows,smoke()))
        rows=good_rows();rows[1]['agent_patch']=''
        self.assertFalse(promotion_candidate(rows,smoke()))
        rows=good_rows();rows[0]['resolved']=False
        self.assertFalse(promotion_candidate(rows,smoke()))
        self.assertFalse(promotion_candidate(good_rows()[:-1],smoke()))
    def test_manual_review_is_separate_required_gate(self):
        rows=good_rows()
        self.assertFalse(readiness(rows,smoke()))
        self.assertTrue(readiness(rows,smoke(),review()))
        for field in ('patch_relevant','behavior_check_relevant','final_diff_reviewed'):
            r=review();r['tasks'][3][field]=False
            self.assertFalse(readiness(rows,smoke(),r))
        r=review();r['tasks'].pop()
        self.assertFalse(review_complete(r))
    def test_review_existing_uses_saved_outputs_no_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for item in good_rows():
                (out/('candidate_'+item['instance_id']+'.json')).write_text(json.dumps(item))
            (out/'real_smoke').mkdir()
            (out/'real_smoke'/'smoke.json').write_text(json.dumps(smoke()))
            rf=out/'review.json';rf.write_text(json.dumps(review()))
            self.assertTrue(review_existing(out,rf)['ready_for_15_tasks'])
    def test_unexpected_nested_tool_shape_fails_closed(self):
        calls=trace_calls({'events':[{'tool':{'name':'run_command'},
                                      'arguments':{'command':CHECK}},
                                     *trace()]})
        self.assertEqual([name for name,_,_ in calls],
                         ['edit_file','run_command','submit_patch'])

    def test_nested_evaluator_trace(self):
        self.assertEqual([name for name,_,_ in trace_calls({'events':trace()})],
                         ['edit_file','run_command','submit_patch'])

if __name__=='__main__':unittest.main()
