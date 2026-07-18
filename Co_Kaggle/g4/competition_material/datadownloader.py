import kagglehub
from kagglehub.exceptions import UnauthenticatedError


COMPETITION = "ai-agent-security-multi-step-tool-attacks"


def download_competition() -> str:
    """Download the competition data, logging in interactively if necessary."""
    try:
        return kagglehub.competition_download(COMPETITION)
    except UnauthenticatedError:
        print("Kaggle authentication is required.")
        print(
            "Create an API token at https://www.kaggle.com/settings, "
            "then paste it at the hidden prompt below."
        )
        kagglehub.login()
        return kagglehub.competition_download(COMPETITION)


if __name__ == "__main__":
    path = download_competition()
    print("Path to competition files:", path)
