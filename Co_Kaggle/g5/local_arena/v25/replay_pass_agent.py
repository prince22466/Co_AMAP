"""Verification fixture for the replay runner's two-argument agent interface."""

def agent(obs, config):
    assert config["turnsPerDay"] == 24
    assert obs["player"] in (0, 1)
    assert "farms" in obs and "private" in obs
    assert len(obs["private"]["inventories"]) == 1 + len(obs["farms"][obs["player"]]["hands"])
    return {"farmer": ["PASS"], "hands": [], "market": []}
