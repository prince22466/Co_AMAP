---
name: candidate
description: Edit a staged candidate, check baseline failure and repaired behavior, then publish a validated snapshot. Fixed script scripts/gate.py; optional args selects an operation.
---

Always use run_skill_script(skill_name="candidate", file_path="scripts/gate.py", args=...).
With no args, show status. Operations: search(text), read(path,start), edit(path,old,new),
write(path,content), check(module,probe,test optional), publish(). Paths belong in args.path.
No arbitrary shell execution. Edits need no assertion. Native read_file sees the live
baseline; candidate read sees staged edits. publish reruns the stored probe and only then
copies the candidate to the live workspace. Call native submit_patch afterward.
