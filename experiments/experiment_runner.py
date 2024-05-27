import torch
from lightning.pytorch.callbacks import DeviceStatsMonitor
from architecture.model import TimeSeriesRepresentationModel, model_dict
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
import lightning.pytorch as pl
import pandas as pd
import numpy as np
import os
import yaml
import time
import json
import hashlib
import random
from typing import Literal

torch.set_float32_matmul_precision("medium")


class ExperimentRun:

    def __init__(self, dataset, config_str: str,
                 task: Literal["finetune_imputation", "finetune_forecasting", "finetune_classification"]):

        self.dataset = dataset
        self.task = task
        self.config_str = config_str
        self.config_pretrain = yaml.safe_load(open(f"experiments/configs/{config_str}_pretrain.yml"))
        self.config_finetune = self.config_pretrain.update(yaml.safe_load(open(f"experiments/configs/{config_str}_finetune.yml")))

        self.run_id = self.generate_run_id()
        self.exp_id = None

    def generate_run_id(self):
        h = hashlib.md5()
        h.update(str(self.dataset).encode())
        h.update(self.task.encode())
        h.update(json.dumps(self.config_pretrain, sort_keys=True).encode())
        h.update(json.dumps(self.config_finetune, sort_keys=True).encode())
        return h.hexdigest()

    def run(self, dry_run=True, **kwargs):
        try:
            self.pre_train(dry_run=dry_run)

            self.fine_tune(dry_run=dry_run, **kwargs)

        except Exception as e:
            if dry_run:
                raise e
            print(f"Error: {e}")
            time.sleep(10)


    def create_log_path(self, phase, checkpoint_path="model_checkpoints"):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        return os.path.join(checkpoint_path, self.run_id, phase)

    def set_seeds(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def pre_train(self, dry_run=True):

        base_config = self.config_pretrain
        base_config["dry_run"] = dry_run
        base_config["task"] = self.task
        self.set_seeds(base_config["shuffle_seed"])

        print(f"Start pretrain ({self.config_str}): run_id: {self.run_id} \nconfig:{str(base_config)}")

        try:

            log_path = self.create_log_path("pretrain")
            metrics = self.run_config_pre_train(base_config, path=log_path)

            print(f"Finished pretrain ({self.config_str}): run_id: {self.run_id} \n{str(metrics)}")

        except Exception as se:
            if dry_run:
                raise se
            print(f"ERROR: {se}")

    def run_config_pre_train(self, config, path: str):
        dataset = self.dataset(config)

        train_loader, validation_loader, test_loader = dataset.get_pretrain_loader()

        model = TimeSeriesRepresentationModel(config=config, scaler=dataset.scaler)

        early_stopping = EarlyStopping(monitor="loss",
                                       min_delta=config["earlystopping_min_delta"],
                                       patience=config["earlystopping_patience"], verbose=True, mode="min")
        checkpoint_callback = ModelCheckpoint(dirpath=path, save_top_k=1, monitor="loss")
        logger = TensorBoardLogger(path, name="log")
        logger.log_hyperparams(config)

        trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=300, log_every_n_steps=20,
                             callbacks=[early_stopping] + [checkpoint_callback]
                             if not config["dry_run"] else [DeviceStatsMonitor(cpu_stats=True)],
                             default_root_dir=path, logger=logger, profiler="simple" if config["dry_run"] else None)

        tuner = Tuner(trainer)
        tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=validation_loader, min_lr=1e-4,
                      max_lr=0.01)

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

        trainer.test(ckpt_path="best", dataloaders=test_loader)

        mets = {k: v.item() for k, v in trainer.logged_metrics.items()}
        mets.update({"version": trainer.logger.version})
        if "loss" not in mets.keys():
            raise IOError("No loss in metrics")

        logger.finalize("success")
        return mets

    def fine_tune(self, dry_run=True, **kwargs):
        checkpoints = [chkp for chkp in os.listdir(self.create_log_path("pretrain")) if chkp.endswith("ckpt")]
        if len(checkpoints) == 0:
            raise IOError("Could not find checkpoints of pretrained model")
        checkpoint = checkpoints[0]

        base_config = self.config_finetune

        print(f"Start finetune ({self.config_str}): run_id: {self.run_id} \nconfig:{str(base_config)}")

        self.set_seeds(base_config["shuffle_seed"])

        try:

            metrics = self.run_config_fine_tune(base_config, checkpoint)
            best_metrics = pd.concat([pd.Series(entry) for entry in metrics], axis=1)

            print(f"Finished finetune ({self.config_str}): run_id: {self.run_id}:\n{str(best_metrics)}")

        except Exception as e:
            if dry_run:
                raise e
            print(f"ERROR: {e}")

    def run_config_fine_tune(self, config, checkpoint):
        dataset = self.dataset(config)
        path = self.create_log_path(config, "finetune")

        metrics = []
        split_num = 0
        for train_loader, val_loader, test_loader in dataset.cv_split():
            model = model_dict[self.task].load_from_checkpoint(
                self.create_log_path(self.config_pretrain, "pretrain") + f"/{checkpoint}",
                config=config, scaler=dataset.scaler)

            early_stopping = EarlyStopping(monitor="loss",
                                           min_delta=config["earlystopping_min_delta"],
                                           patience=config["earlystopping_patience"], verbose=True, mode="min")
            checkpoint_callback = ModelCheckpoint(dirpath=path, save_top_k=1,
                                                  monitor="mse_missing" if self.task != "finetune_classification" else "f1_classification")
            logger = TensorBoardLogger(path, name="log")


            trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=300, log_every_n_steps=20,
                                 callbacks=[early_stopping] + (
                                     [checkpoint_callback] if not config["dry_run"] else []),
                                 default_root_dir=path, logger=logger)

            tuner = Tuner(trainer)
            tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader, min_lr=1e-4,
                          max_lr=0.01)

            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            trainer.test(ckpt_path="best", dataloaders=test_loader)
            mets = {k: v.item() for k, v in trainer.logged_metrics.items()}
            mets.update({"version": trainer.logger.version})

            metrics.append(pd.Series(mets, name=f"split_{split_num}"))
            split_num += 1

        return metrics

