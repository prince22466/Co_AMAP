https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/evaluation

# Evaluation

You must provide a zip archive (submission.zip) that contains your Agent Config, comprising system prompts, custom tools, skills, and LoRA adapters. An agent.yaml file must be located at the root of the archive.

```text
submission.zip
├── agent.yaml                  # REQUIRED: Root agent config
├── configs/
│   └── sampling.yaml           # Optional: Generation parameters loaded via !include
├── prompts/
│   ├── system.md               # Optional: System instructions loaded via !include
│   └── analyzer.md
├── sub_agents/
│   └── code_analyzer.yaml      # Optional: Sub-agent or AgentTool YAML configurations
├── adapters/                   # Optional: Fine-tuned PEFT LoRA adapters or model weights
│   ├── main_lora/
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   └── tool_lora/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
└── skills/                     # Optional: ADK Skill directories
    └── repo_navigation/
        ├── SKILL.md
        ├── scripts/            # Python or bash scripts executed in a sandbox
        └── resources/          # Domain knowledge markdown files
```

The config language follows the Google ADK Agent Config specification with additional restrictions to prevent code execution outside of a sandbox. The submissions themselves are compiled into ADK agents to be evaluated.

## Issue Scoring

Similar to SWE-Bench, your agent's submitted patches are evaluated by a PASS/FAIL metric. For each issue, the submitted patch is applied to that issue's code repository and that issue's validation tests are run. Your submission's overall score is the percentage of patched repositories that pass the validation tests.

Your agent has a limit of 12 hours to submit patches for all tasks, inclusive of sandbox setup time, but excluding time for patch validation. You may, but are not required, to set per-task time limits in your submission's eval_config.yaml.
