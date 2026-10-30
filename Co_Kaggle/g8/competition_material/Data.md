# Data and Submission Interface

**Primary source:** [Kaggle — Data](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/data)  
**Cross-check:** [Official starter kit](https://github.com/smart-buildings-2026/starter-kit) and its [transfer-policy template](https://github.com/smart-buildings-2026/starter-kit/blob/main/smart_buildings/baselines/transfer_policy.py)

## 1. What you build

This is a **building-control competition**, not a tabular prediction task. There is **no test CSV with labels to predict**.

Submit a **Python control policy** that, at every **5-minute** simulation step:

1. Receives the current building's observation vector.
2. Chooses the HVAC equipment to run.
3. Returns discrete equipment commands and continuous control setpoints.

The objective is a single policy that works on unseen building layouts and weather conditions. See [Evaluation.md](Evaluation.md) for the reward and leaderboard metric.

## 2. How to submit on Kaggle

The competition uses **Kaggle Notebook submissions**, not a manually uploaded prediction CSV. The organizer's **TransferPolicy Notebook** on the competition's [Code tab](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/code) is the starting example.

In the Kaggle inference environment, the policy subclass follows this interface:

- Inherit from \`smart_buildings_inference_server.Policy\`.
- Implement \`__init__(building_meta_data, action_spec)\`.
- Implement \`get_action(obs)\`, returning an action dictionary.

The action dictionary has two entries:

| Key | Type | Meaning |
| --- | --- | --- |
| \`discrete_action\` | Integer array (typically \`int32\`) | AHU and hot-water-system on/off commands |
| \`continuous_action\` | Floating-point array (typically \`float32\`) | Normalized HVAC setpoints |

The notebook runs a public example scenario as a check; the evaluation gateway runs the **same submitted notebook** against hidden scenarios. The hidden evaluation does not require changing the policy code.

**Important interface distinction:** The **local starter-kit** \`TransferPolicy\` class implements the same constructor and action method but **does not need to inherit from a base class**. When submitting on Kaggle, follow the organizer's **Kaggle inference-server notebook**, not the local class definition verbatim.

The competition's [code requirements](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/overview) specify an execution limit of **9 hours** for CPU or GPU notebooks and **disabled internet access**.

## 3. Training data and simulator

There is no fixed labeled training dataset. The organizers provide:

- A Gym-compatible simulator built on an extended version of Google's **Smart Buildings Control Suite**.
- A procedural building generator to produce additional training buildings.
- **30 shipped training scenarios**, based on **6 buildings** and **5 weather stations**.
- Baseline controllers, an evaluation runner, and a local score calculator.

The shipped training scenarios pair each building with one 30-day weather strip from each of five stations, as specified in [\`training_set.yaml\`](https://github.com/smart-buildings-2026/starter-kit/blob/main/smart_buildings/training_set.yaml). These are **30 selected building–weather pairs**, not every possible cross-product of every available weather strip.

The stations used in the shipped scenarios are \`fairbanks\`, \`denver\`, \`atlanta\`, \`phoenix\`, and \`singapore\`. Each weather strip is a distinct 30-day window of recorded weather. Participants can generate additional training scenarios.

The **hidden evaluation buildings and weather** are not provided during the competition. Building metadata is given to the policy at evaluation time, so the policy must generalize to new configurations.

### Building variation

The procedural generator varies:

- Floor count (**2–4**), footprint, perimeter, and floor layout
- Zone and occupancy configuration
- Six building-envelope material properties
- Air-handler units (**AHUs**), with at most **10 zones per AHU**
- Zone-level variable-air-volume (**VAV**) boxes

Each zone has one VAV box, and each building has a central hot-water system.

## 4. Observation space

At each step, \`get_action(obs)\` receives a **flat observation vector**. Its length depends on the building, so do **not** hard-code observation offsets.

| Scope | Observation fields |
| --- | --- |
| Each zone | Air temperature; heating and cooling setpoints; damper position; reheat valve position; airflow setpoint |
| Each AHU | Supply-air temperature; static pressure; measured and setpoint airflow; recirculation; cooling-request count; fan status |
| Hot-water system | Supply-water temperature (sensed and setpoint) |
| Building and environment | Occupancy count; current/upcoming comfort-schedule flags; outdoor-air temperature; cyclic time-of-day and day-of-week encodings |

The \`building_meta_data\` dictionary provides the **observation field names**, action indices, zone-to-AHU assignments, floor plan, comfort schedule, and normalization constants.

## 5. Action space

Let:

- \`K\` = number of AHUs.
- \`V\` = number of zones (and VAV boxes).

The action is a **dictionary with variable-length discrete and continuous arrays**:

| Component | Range | Shape | Controls |
| --- | --- | --- | --- |
| \`discrete_action\` | \`{0, 1}\` | \`(K + 1,)\` | One on/off command per AHU, plus one for the hot-water system |
| \`continuous_action\` | \`[-1, 1]\` | \`(3K + 2 + 2V,)\` | Three setpoints per AHU, two for hot water, and two per zone |

### Continuous action breakdown

| Equipment | Number | Controlled variables |
| --- | --- | --- |
| AHUs | \`3K\` | Supply-air temperature, static pressure, recirculation |
| Hot-water system | \`2\` | Supply-water temperature, differential pressure |
| VAV boxes | \`2V\` | Damper position, reheat valve position |

**Zero is the midpoint of a normalized continuous range, not the off command.** Use the action-index mappings from \`building_meta_data\` rather than assuming the fields appear in contiguous groups or a fixed order. The official local template documents \`temp_indices\`, \`pressure_indices\`, \`recirculation_indices\`, \`water_indices\`, and \`zone_action_indices\`.

### Simulator-imposed ventilation limits

The simulator enforces these physical limits:

- A running AHU maintains a minimum static pressure.
- Recirculation is capped at **70%**.
- Every VAV damper remains at least **10%** open.

Turning off all airflow is therefore not a viable universal energy-saving strategy.

## 6. Local development

The [starter kit](https://github.com/smart-buildings-2026/starter-kit) includes the simulator, generator, training scenarios, reference policies, and tools for local evaluation.

\`\`\`bash
pip install -e .

# Inspect a public training scenario and its spaces.
python -m smart_buildings.run_environment --building building_train_1 --weather fairbanks_1

# Evaluate the unmodified starter policy on all 30 shipped scenarios.
python -m smart_buildings.evaluate
\`\`\`

The local policy implementation is in \`smart_buildings/baselines/transfer_policy.py\`. The supplied example returns an all-zero action for compatibility testing, **not** as a recommended competitive strategy.

## 7. Repository data availability

This GitHub repository currently contains the Markdown competition documents but **does not contain** a checked-in \`Co_Kaggle/g8/competition_material/competition_data/\` directory. Large simulator inputs should be obtained from the **official starter kit** or inspected in the Kaggle environment; their presence on a local computer is not evidence that they are committed here.

**Verification note (2026-10-30):** The observation/action shapes, training-scenario definitions, and local policy method signatures above were checked against the organizer's public starter kit. The Kaggle **Data** page did not expose its complete text to automated retrieval during this review, so Kaggle-only notebook/gateway details should be rechecked against the live organizer example before a submission.
