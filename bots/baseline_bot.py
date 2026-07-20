"""Local comparison bot using the current submission's semantic base policy."""

from runtime import run_bot
from strategies.semantic_potential import SemanticPotentialStrategy


if __name__ == "__main__":
    run_bot(SemanticPotentialStrategy)
