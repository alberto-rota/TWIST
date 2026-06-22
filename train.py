#!/usr/bin/env python
"""Training entry point.

    python train.py                       # uses config/train.yaml
    python train.py config/smoke.yaml     # any config
    python train.py -b                    # boot: tiny smoke run
    python train.py --DATASETS.KUBRIC.MAX_POINTS=512   # override any field

For W&B sweeps use sweep_agent.py (see config/sweep.yaml).
"""

from main import run_pipeline

if __name__ == "__main__":
    run_pipeline(mode="train")
