Link:https://www.kaggle.com/competitions/neurips-2026-smart-buildings/data


downloaded data is in competition_material\competition_data,
embeddings,graphs,snapshots are too large to download. which should be inspected in kaggle.


# Dataset Description

What You're Building
You are not predicting labels, and there is no test CSV. You submit a control policy: a Python class that, every 5 minutes, reads a building's sensor observations and outputs HVAC control actions. We run your policy inside the simulator on held-out buildings and score how well it balances occupant comfort, energy cost, and carbon emissions (see the Evaluation tab).

How to Submit
You submit a Kaggle notebook, nothing else. A working example is on the Code tab: TransferPolicy Notebook. Copy it, replace the control logic in get_action, and submit.

Your class subclasses smart_buildings_inference_server.Policy and implements __init__(building_meta_data, action_spec) and get_action(obs).
When you run the notebook yourself, it plays one public example scenario so you can check it works.
When you submit, the same notebook is re-run against the hidden test scenarios. You do not change any code.
To train and score locally, use the starter kit on GitHub. It has the same simulator, the 30 training scenarios, the baselines, and a local scorer.

Overview
This competition has no static dataset. You get a building simulator and a procedural building generator, so you can create as much training data as you want. All buildings are simulated with an extended version of Google's open-source Smart Buildings Control Suite, and each one is a Gym-compatible reinforcement learning environment.

Buildings
The generator varies each building's:

Number of floors (2–4), width, length, and perimeter shape
Room layout and occupancy
Envelope construction (six material properties)
Number of air handler units (AHUs), which is set by footprint (at most 10 zones per AHU)
Every zone has its own VAV box. Each building has one hot-water system.

Weather
Weather covers a range of climates. The training set pairs 6 buildings with 30 weather strips, named <station>_<n>. Each strip is a different 30-day window of recorded data.

Your score comes only from the hidden test set, not the 30 training scenarios. The test buildings come from the same generator and parameter ranges as the training buildings, but they are evaluated under different weather than the training set. Your policy must generalize to both new buildings and new weather conditions.

Test buildings and test weather are never released during the competition. Your policy receives each test building's metadata only at runtime, inside the evaluation sandbox. You may generate as many additional buildings and scenarios as you like for training.

Observation Space
At each 5-minute step, your policy receives a flat vector (its length depends on the building) containing:

Per zone: air temperature, current heating and cooling setpoints, damper position, reheat valve position, and airflow setpoint
Per AHU: supply-air temperature, static pressure, airflow (sensed and setpoint), recirculation, cooling-request count, and fan run status
Hot-water system: supply water temperature (sensed and setpoint)
Building-wide: occupancy count, comfort-schedule flags (in comfort mode now or soon), outside air temperature, and time of day and day of week (cyclically encoded)
A building metadata dict is passed to your policy's constructor. It contains action indices, observation field names, the zone-to-AHU assignment, the floor plan, the schedule, and normalization constants.

Action Space
The action is hybrid: some parts are discrete and some are continuous.

Discrete ({0, 1}, shape ((K+1,))): an on/off run command for each AHU and for the hot-water pump
Continuous ([-1, 1], shape ((3K+2+2V,))):
Per AHU: supply-air temperature, static pressure, and recirculation
Hot-water system: supply-water temperature and differential pressure
Per zone: damper position and reheat valve position
(K) is the number of AHUs and (V) is the number of zones. Zero is the middle of each continuous range, not "off". Read the index lists in the metadata rather than assuming the order of the action vector.

The simulator enforces ventilation minimums: a running AHU keeps a minimum static pressure, recirculates at most 70%, and every damper stays at least 10% open. Shutting off all airflow is not a viable strategy.

The full file guide, setup instructions and local scoring tools are in the starter kit README on GitHub.