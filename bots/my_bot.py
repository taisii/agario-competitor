from runtime import run_bot
from strategies.replay_distilled import ReplayDistilledStrategy


def main() -> None:
    run_bot(ReplayDistilledStrategy)


if __name__ == "__main__":
    main()
