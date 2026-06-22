#!/usr/bin/env python
"""W&B sweep agent.

A sweep launched from ``config/sweep.yaml`` runs this file once per trial. W&B
injects the trial's hyper-parameters as ``wandb.config`` (a flat ``{KEY: value}``
dict, including dotted nested keys like ``DATASETS.KUBRIC.MAX_POINTS``), which we
hand straight to ``run_pipeline`` -- the exact same path as ``python train.py``.

    wandb sweep config/sweep.yaml          # -> SWEEP_ID
    wandb agent <entity>/<project>/<id>    # runs this agent
"""

import wandb

import main


def sweep_agent() -> None:
    with wandb.init():
        main.run_pipeline(mode="train", config=dict(wandb.config))


if __name__ == "__main__":
    sweep_agent()
