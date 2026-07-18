# AI Agent Security - Multi-Step Tool Attacks: 
https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/overview

Overview
In this competition, hosted by OpenAI, Google, and IEEE, you will build an attack algorithm that stress-tests tool-using AI agents in a deterministic offline benchmark. Your goal is to find multi-step attack paths that move an agent from untrusted inputs to unsafe actions, then return replayable findings that the evaluator can verify.

This is not a prompt-writing challenge. It is an algorithmic red-teaming challenge for agent security.

Description
Modern AI systems increasingly operate through tools. They can search the web, inspect files, write data, send emails, make HTTP requests, and run commands. These capabilities make agents more useful, but they also make failures more consequential. A harmful instruction hidden in an email, webpage, file, or prior interaction may influence later tool calls and cause the agent to leak data, modify files, misuse authority, or take actions the user never intended.

Most AI safety evaluations still focus on single prompts or isolated jailbreaks. Those tests are useful, but they do not fully capture the risks of tool-using agents. Realistic failures often emerge across multiple steps: an agent reads untrusted content, stores or summarizes it, plans a follow-up action, calls one or more tools, and only later crosses a security boundary. These behaviors are hard to measure with static prompt tests, hard to reproduce with manual red teaming, and hard to compare across research groups without a shared benchmark.

This competition introduces a deterministic offline benchmark for agent-security evaluation. Participants will submit an attack.py file that implements an attack search algorithm using the competition SDK and Gym-style environment. Your algorithm will interact with a sandboxed tool-using agent, explore possible attack paths, and return replayable attack candidates. The evaluator will replay those attack moves in a fresh environment, recompute the trace, and score against the security predicates.

The benchmark focuses on unsafe action patterns that matter for real tool-using systems, including:

Exfiltration: causing an agent to leak sensitive information through an external channel.  
Untrusted-to-action: causing content from an untrusted source to trigger a dangerous action.  
Destructive write: causing an agent to modify, overwrite, or delete protected data.  
Confused deputy behavior: causing an agent to misuse its authority on behalf of an attacker or unintended context.  
Your solution may use any attack discovery strategy that fits the competition rules. 
  
*** Promising approaches may include prompt search, fuzzing, heuristic search, evolutionary algorithms, state-space exploration, trace-guided mutation, novelty search, Go-Explore-style archive methods, LLM-assisted candidate generation, or hybrid systems that learn from prior traces. *** 
Successful submissions will need to find high-impact failures, produce diverse replayable attack paths, and use the evaluation budget efficiently.
  
The environment is designed to make agent-security research more reproducible. Instead of attacking live systems, submissions operate in a controlled offline sandbox with fixture-backed tools. This lets participants test realistic tool-use risks while keeping evaluation deterministic, comparable, and safe for a public leaderboard.

By participating, you will help define how the field measures security for tool-using AI agents. The strongest submissions will do more than climb the leaderboard: they will produce reusable attack algorithms, standardized failure cases, and practical insight into how multi-step agent failures happen. Those findings can help researchers and builders compare defenses, identify weak points earlier, and develop safer agent systems before deployment.


Getting Started
To get started, following this starter notebook, initialize an attack template, and run a local smoke test against the deterministic environment.

Your public submission should provide an attack.py file defining an AttackAlgorithm class. *** The algorithm should interact with the environment, search for unsafe action traces, and return replayable attack candidates.***  
The hosted evaluator will validate and score those candidates by replaying them, rather than trusting attacker-provided metadata.


What Makes This Different
This benchmark is different from classic jailbreak evaluations in three ways.

First, the target system can use tools, so the relevant failure mode is not only harmful text, but unsafe action.

Second, attacks can unfold over multiple steps, where untrusted content, planning, memory, and tool use interact.

Third, scoring is replay-validated. Submissions must discover attack paths that reproduce under the evaluator, making results more comparable and useful for future research.
