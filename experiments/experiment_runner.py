import torch
from lightning.pytorch.callbacks import DeviceStatsMonitor
from architecture.model import model_dict
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from lightning.pytorch.tuner import Tuner
import lightning.pytorch as pl
import numpy as np
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import hashlib
import random
from typing import Literal
from data_provider.data_factory import data_provider


torch.set_float32_matmul_precision("medium")


class ExperimentRun:

    def __init__(self, experiment, config: dict, phase: Literal["pretrain", "finetune", "downstream"]):

        self.experiment = experiment
        self.base_config = experiment.config
        self.exp_run_config = config
        self.phase = phase

        self.run_id = self.generate_run_id()

    def generate_run_id(self, phase=None):
        h = hashlib.md5()
        h.update(self.experiment.task.encode())
        h.update((phase or self.phase).encode())
        h.update(self.experiment.get_exp_run_id().encode())
        h.update(json.dumps(self.exp_run_config, sort_keys=True).encode())
        h.update(json.dumps(self.base_config, sort_keys=True).encode())

        return h.hexdigest()

    def create_log_path(self, checkpoint_path="model_checkpoints", phase=None, run_id=None):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        return os.path.join(checkpoint_path, self.experiment.get_exp_run_id(), phase or self.phase,
                            run_id or self.run_id)

    def set_seeds(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def run(self, dry_run=False, pre_train_run_id=None, log_path_checkpoint=None):
        checkpoint = None
        exp_run_config = self.exp_run_config

        if self.phase == "finetune":
            checkpoints = [chkp for chkp in os.listdir(log_path_checkpoint or self.create_log_path(phase="pretrain", run_id=pre_train_run_id))
                           if chkp.endswith("ckpt") and "epoch" in chkp]
            if len(checkpoints) == 0:
                raise IOError("Could not find checkpoints of pretrained model")
            checkpoint = os.path.join(log_path_checkpoint or self.create_log_path(phase="pretrain", run_id=pre_train_run_id), checkpoints[0])

        exp_run_config["dry_run"] = dry_run
        exp_run_config["task"] = self.experiment.task
        exp_run_config["phase"] = self.phase
        exp_run_config["num_workers"] = exp_run_config.get("num_workers", 2) if not dry_run else 0
        self.set_seeds(exp_run_config["shuffle_seed"])

        print(f"Start {self.phase} ({self.experiment.get_exp_run_id()}): ({self.run_id}) \nconfig:{str(exp_run_config)}")

        try:

            log_path = self.create_log_path()
            start_time = time.time()
            metrics = self.run_config(exp_run_config, path=log_path, checkpoint=checkpoint)
            elapsed_time = time.time() - start_time

            result_dict = {
                "exp_id": self.experiment.exp_id,
                "codename": self.experiment.codename,
                "run_id": self.run_id,
                "phase": self.phase,
                "timestamp": datetime.now(tz=ZoneInfo("Europe/Berlin")),
                "metrics": metrics,
                "elapsed_time_minutes": elapsed_time / 60,
                "config": exp_run_config,
                "log_path": log_path
            }
            # if you want to save for later evaluation, implement logic here.

            print(f"Finished {self.phase} ({self.experiment.get_exp_run_id()}): ({self.run_id})\n{str(metrics)}")

        except Exception as se:
            if dry_run:
                raise se
            print(f"ERROR: {se}")

    def run_config(self, config, path: str, checkpoint=None):
        if self.phase == "finetune" and checkpoint is None:
            raise IOError("No Checkpoint set")

        train_data, train_loader = data_provider(config, flag="train")
        validation_data, validation_loader = data_provider(config, flag="val")
        test_data, test_loader = data_provider(config, flag="test")

        if self.phase == "pretrain":
            model = model_dict["pretrain"](config=config)

        elif self.phase == "finetune":
            model = model_dict[self.experiment.task].load_from_checkpoint(checkpoint, config=config)
        else:  # downstream
            model = model_dict[self.experiment.task](config=config)

        model.float()

        print(f"Trainable Parameter: {ModelSummary(model).trainable_parameters}")

        early_stopping = EarlyStopping(monitor="loss",
                                       min_delta=config["earlystopping_min_delta"],
                                       patience=config["earlystopping_patience"], verbose=True, mode="min")
        checkpoint_callback = ModelCheckpoint(dirpath=path, save_top_k=1, monitor="loss", every_n_epochs=1)
        # logger = TensorBoardLogger(path, name="log")
        # logger.log_hyperparams(config)

        mlflow = MLFlowLogger(experiment_name=self.experiment.get_exp_run_id(), run_name=self.phase + "_" + self.run_id, tracking_uri="http://localhost:5000")
        mlflow.log_hyperparams(config)
        trainer = pl.Trainer(devices=1, accelerator="gpu", precision="16-mixed",
                             callbacks=([early_stopping, checkpoint_callback])
                             if not config["dry_run"] else [DeviceStatsMonitor(cpu_stats=True)],
                             default_root_dir=path, logger=[mlflow], profiler="simple" if config["dry_run"] else None)

        if self.phase != "finetune":
            tuner = Tuner(trainer)
            tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=validation_loader, min_lr=1e-4,
                          max_lr=0.01)

            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
            model.training = False
            trainer.test(ckpt_path="best", dataloaders=test_loader)
        else:
            trainer.test(model, dataloaders=test_loader)


        mets = {k: v.item() for k, v in trainer.logged_metrics.items()}
        mets.update({"version": trainer.logger.version})
        if "loss" not in mets.keys():
            raise IOError("No loss in metrics")

        model_summary = ModelSummary(model)
        mets.update({
            "trainable_parameter": model_summary.trainable_parameters
        })

        [log.finalize("success") for log in trainer.loggers]

        return mets


