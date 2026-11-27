"""Optimize an OptiMind prompt with grounded diagnostics and verified practice problems."""
import sys
from experiments.gepa.optimize import main as optimize

def main(argv=None, checkpoint=None, deadline=None):
    return optimize(argv, checkpoint, deadline, decompevolve=True)


if __name__ == "__main__":
    sys.exit(main())
