from typing import Literal, Optional

import numpy as np
import yaml
import time
from experiments.scheduler import next_run


class Experiment:

    def __init__(self, exp_id: str, codename: str,
                 config_str: str,
                 task: Literal["imputation", "forecasting", "classification"],
                 mode: Literal["downstream", "pre-trained"],
                 hyperparameters_pretrain: Optional[dict] = None, 
                 hyperparameters_finetune: Optional[dict] = None,
                 hyperparameters: Optional[dict] = None):
        self.exp_id = exp_id
        self.codename = codename
        self.config_str = config_str
        self.task = task
        self.mode = mode
        self.config = yaml.safe_load(open(f"experiments/configs/{config_str}.yml"))

        if mode == "pre-trained":
            self.hyperparameters_pretrain = hyperparameters_pretrain
            self.hyperparameters_finetune = hyperparameters_finetune
            
        elif mode == "downstream":
            self.hyperparameters = hyperparameters
        else:
            raise IOError(f"Unknown mode: {mode}")

    def get_exp_run_id(self):
        return f"{self.exp_id}_{self.codename}_{self.mode}"

    
    def run_downstream(self, dry_run=False):
        scheduler = next_run(experiment=self, phase="downstream")
        for experiment_run, hyper_config in scheduler:

            try:
                experiment_run.run(dry_run=dry_run)
            except Exception as e:
                if dry_run:
                    raise e
                print(f"Error: {e}")
                scheduler.throw(e)
                time.sleep(5)
