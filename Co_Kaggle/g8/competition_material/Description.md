# NeurIPS 2026 Smart Buildings Competition — Overview

**Challenge:** Train an efficient HVAC control system that generalizes to buildings it has not seen before.

**Official source:** [Kaggle competition overview](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/overview)

## Competition at a glance

In the **Smart Buildings Challenge**, participants develop a control policy for buildings with different sizes, floor plans, weather, construction materials, and HVAC configurations.

The goal is to improve energy efficiency and reduce carbon emissions **without sacrificing occupant comfort**. Crucially, the policy must transfer to previously unseen buildings rather than merely perform well in one familiar environment.

## Why this challenge matters

According to the competition description:

- Buildings account for nearly **40% of carbon emissions in the United States**.
- HVAC systems consume **40–60% of a building's energy** and **15% of global energy**.
- Many existing building controllers are tailored to one building, and their performance declines when transferred to another.

Although machine learning has advanced, most building-control research still evaluates policies on the same building used for development. This limits the ability to deploy energy-saving policies across campuses, cities, and regions.

## The task: transferable HVAC control

Participants train control policies using simulated buildings. Organizers evaluate the policies on **unseen buildings and weather scenarios** that can differ in:

- Building size and floor layout
- Climate and weather conditions
- Construction and envelope material properties
- HVAC equipment and system configurations

A successful policy must adapt to these differences with minimal per-building configuration.

This is a **sequential control task**, not a conventional fixed-label prediction problem. For the control interface, observations, and submission format, see [Data.md](Data.md). For the exact scoring formula, see [Evaluation.md](Evaluation.md).

## Organizers and potential impact

The competition is hosted by the **NSF AI Institute in Dynamic Systems, Harvard University, Columbia University, the University of Washington, and Kajima Corporation**.

Policies that generalize well could make large-scale building-energy optimization more practical. **Winning solutions may have an opportunity to be tested on three real buildings**, as stated on the official competition page.

## Official resources

- [Competition overview](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/overview)
- [Competition data and submission interface](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/data)
- [Evaluation and scoring](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/overview/evaluation)
- [Organizer's starter kit](https://github.com/smart-buildings-2026/starter-kit)

> **Source fidelity:** This document reorganizes and lightly paraphrases the official Kaggle *Description* section. The numerical motivation and claims about potential real-building testing are attributed to the organizers, not independently established here.
