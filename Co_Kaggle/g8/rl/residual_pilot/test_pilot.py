import unittest
from types import SimpleNamespace
import numpy as np
from pilot import apply, learned, rule, TRAIN, VAL
class PilotTests(unittest.TestCase):
    def test_residuals_respect_off_and_bounds(self):
        v=SimpleNamespace(ahus=[{"run":0,"supply":0,"pressure":1},{"run":1,"supply":2,"pressure":3}])
        cmd={"continuous_action":np.array([.95,.95,.2,.4],dtype=np.float32),"discrete_action":np.array([1,0],dtype=np.int32)}
        self.assertIs(apply(v,cmd,"base"),cmd)
        sat=apply(v,cmd,"sat_up")
        fan=apply(v,cmd,"fan_up")
        np.testing.assert_allclose(sat["continuous_action"],[1,.95,.2,.4])
        np.testing.assert_allclose(fan["continuous_action"],[.95,1,.2,.4])
        np.testing.assert_allclose(cmd["continuous_action"],[.95,.95,.2,.4])
        np.testing.assert_array_equal(fan["discrete_action"],cmd["discrete_action"])
    def test_rule_gates_on_occupancy_schedule_and_demand(self):
        x=np.zeros(12);x[9]=1.
        self.assertEqual(rule(x),"base")
        x[1]=1.;self.assertEqual(rule(x),"sat_down")
        x[8]=2.;self.assertEqual(rule(x),"base")
    def test_learned_policy_abstains(self):
        p={"actions":["fan_up"],"mean":[0.]*12,"scale":[1.]*12,
           "coef":[[-.0001]+[0.]*12],"margin":.0002}
        self.assertEqual(learned(np.zeros(12),p),"base")
        p["margin"]=.00007
        self.assertEqual(learned(np.zeros(12),p),"fan_up")
    def test_split_disjoint(self):
        self.assertFalse({s.split("/")[0] for s in TRAIN}&{s.split("/")[0] for s in VAL})
        self.assertFalse(any(s.startswith("building_train_8/") for s in TRAIN+VAL))
if __name__=="__main__":unittest.main()
