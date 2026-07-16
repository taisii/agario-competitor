from entries.common import run_strategy
from strategies.semantic_offensive_beam import SemanticOffensiveBeamStrategy


if __name__ == "__main__":
    run_strategy(SemanticOffensiveBeamStrategy())
