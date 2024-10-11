from sklearn.model_selection import ParameterGrid
from typing import Literal

from experiment_runner import ExperimentRun
import random


def next_run(experiment, phase: Literal["pretrain", "finetune", "downstream"]):

    if phase == "downstream":
        hyper_paras = experiment.hyperparameters
    elif phase == "pretrain":
        hyper_paras = experiment.hyperparameters_pretrain
    elif phase == "finetune":
        hyper_paras = experiment.hyperparameters_finetune
    else:
        raise IOError(f"Unknown phase {phase}")

    configs = list(ParameterGrid(param_grid=hyper_paras))
    random.shuffle(configs)

    for config in configs:

        exp_run_config = experiment.config.copy()
        exp_run_config.update(config)
        exp_run = ExperimentRun(experiment, exp_run_config, phase)

        try:
            yield exp_run, config
        except Exception as e:
            print(f"Error during experiment {experiment.exp_id}: {e}")
