1st submission on aug 20th. game ends on Sep 30 2026.

before v16, simply use codex to generate next version by using self generated seed and play against current version for evaluation.
v16-v18, using codex and game history(loss ones) for generating next version
form v19, start to use rl for training.

system workflow structure for notebooks starting from v16:

notebook | structure | comments |






v22 is new reorganized version, which has clear structure 
v22 is submitted to kaggle on Sep30th, and loses terribly(as of writing, Oct 1st 2026, gets only 439.1), its rl part(for worker actions) is based on best.pt(commit sha:ceb2acbf3ee14a210c7ee3a01ac2a0c93c6e688f, commited on Sep 30th)

--------------------------------------------------------------------------------------------------------------------------------------
what to do afterwards,
keep training rl model for worker action,
study reward design to make training faster and make agent behavior as intended,
study rl methods ppo, q, deep rl, agentic rl training,
setup systemic, engineered way to generate in-time feedback on how the training system works(aka, effective observality and reaction to improve training system),
training platform to faciliate distirbuted rl trainig,
training platform to faciliate distirbuted rl trainig + distributed ml/dl trainng,
