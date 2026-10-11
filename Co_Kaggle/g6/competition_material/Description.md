# Google - The Gemma 4 Developer Agent Competition

https://www.kaggle.com/competitions/gemma-4-developer-agent/overview

## Overview

Today’s best coding agents rely on massive API models running in the cloud. What if every developer could rely on an autonomous agent equally capable offline, on consumer hardware?

Post-train an open model into a reliable agent that navigates complex codebases and drafts fixes for real software issues, accelerating developer workflows on everyday hardware.

## Description

Note: there are two active competitions for the Gemma 4 Developer Agent Competition: this competition, and a paper track, where you can document your approach for this prediction competition.

Today’s most capable coding agents rely on internet-connected models that require powerful commercial infrastructure. That puts advanced software engineering out of reach for many developers.

Small local models have made rapid progress, but they still struggle with complex, multi-turn software engineering. Large codebases are hard to navigate, so fixing bugs and adding features reliably is tough for models that fit on a single accelerator.

In this competition, you’ll post-train Gemma 4 to build autonomous software engineering agents designed to amplify developers’ talents. You’ll explore new fine-tuning and reinforcement learning techniques to create agents that can independently navigate repositories and draft solutions, freeing up human engineers to focus on architecture, logic, and review.

Your solution could help put powerful coding assistants in the hands of millions more developers. Sharing how you did it makes advanced coding tools accessible to everyone.

## Model Selection, Budget, and Harness Rules

### Model Selection and LoRA Adapters

At present, we only support the gemma-4-31b-it-qat-w4a16-ct Gemma 4 model variant. You must choose this model for every agent and subagent.

You may, but are not required, to include one or more .safetensors format LoRA adapters with your submission. While you are restricted to a single base model, you may use different adapters for each agent in your submission:

- Place PEFT LoRA directories (containing adapter_config.json and adapter_model.safetensors) inside adapters/<adapter_name>/.
- Reference adapter: <adapter_name> on any LlmAgent in agent.yaml or sub_agents/*.yaml (for example, adapter: main_lora on the root coder agent and adapter: tool_lora on a read-only analyzer AgentTool).

### Tool, Prompt, and Skill Rules

- !include Directives: Resolves file paths relative to the directory of the file containing the tag. For instance, in agent.yaml, !include prompts/system.md loads prompts/system.md.
- Sandboxing: Path traversal outside the submission root (e.g., via ../ or symlinks) is not allowed.
- Allowed Tools: Your agent can only request tools provided by the competition harness or custom subagents defined via agent_tool.
- Skill Structure & Sandboxing: Each skill must be a directory containing a SKILL.md manifest with YAML frontmatter (name: <skill-name>). Scripts executed via run_skill_script run securely inside the competition's persistent Docker container, sharing filesystem access with run_command and debiting execution time against your central budget. Agents can inspect domain knowledge files using load_skill_resource.

The predefined tools available to your agent are:

- `run_command(command: str) -> str` - Executes a shell command in /bin/bash -c inside /workspace.
- `submit_patch() -> str` - Stages untracked file intents (git add -N .) and captures git diff HEAD from /workspace.
- `get_status() -> str` - Returns live budget consumption and patch status.
- `read_file(filepath: str, start_line: int | None = None, end_line: int | None = None) -> str` - Reads a file from /workspace with 1-indexed inclusive line slicing.
- `edit_file(filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> str` - Replaces old_string with new_string in an existing non-empty file inside /workspace.
- `write_file(filepath: str, content: str) -> str` - Creates or overwrites a file at /workspace/<filepath>, automatically creating parent directories (mkdir -p).
- `get_code_neighbors(node: str, edge_type: str | None = None, max_neighbors: int = 50) -> str` - Finds incoming and outgoing neighbors of a symbol (node) in the repository call/dependency graph.
- `search_similar_code(query: str, k: int = 10) -> str` - Finds top-k graph nodes with highest cosine similarity to query in the pre-computed embeddings.
- `get_code_subgraph(nodes: list[str]) -> str` - Extracts the induced subgraph (all nodes and interconnecting edges) for a list of symbols.

See the HARNESS_README.md file in the dataset for detailed information about the submission format and execution environment.

## Upgraded Accelerators

This competition has access to Kaggle's pool of powerful new L4x4 machines! These machines offer 96GB of GPU memory enabling submissions with much larger models. See this page for more on these machines.

What you need to know:

- Quota usage - Due to their limited availability, notebooks with L4x4s consume GPU quota at twice the rate of the older T4x2 and P100 machines. We may increase this rate as necessary to ensure these machines are available for submission scoring.
- Restricted Use - L4s are only available for notebooks attached to this competition. We will build tooling to enforce this if necessary. In the meantime, attempts to circumvent this restriction may be enforced by Kaggle moderation with consequences including team bans from the competition or account bans.
- No Internet - All L4 sessions must have internet disabled.
