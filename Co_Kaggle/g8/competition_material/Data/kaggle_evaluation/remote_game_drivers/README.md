# Game Environment Remote Drivers

Bridges agent containers to game-engine libraries (Gymnasium, PettingZoo, OpenSpiel)
so entire libraries can run on Kaggle's competition infrastructure. Implements
[go/kaggle-game-driver-design-proposal](https://docs.google.com/document/d/1-FjYCMz7VfJ6EuhNGRoKJ_xxlYjVB-qiUe2aC0pRE-I/).

Eventually used by both [Hearth](../../) and
[kaggle-environments](https://github.com/Kaggle/kaggle-environments/); development
happens in the competitions repo until a draft is ready to move to kaggle-environments.

For architecture, layout, how to write an agent, and how to add a library, see
[`agents.md`](agents.md). For setup and test commands, see the top-level
[`AGENTS.md`](../../AGENTS.md).
