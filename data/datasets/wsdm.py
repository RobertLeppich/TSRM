import pickle as pk
from data.datasets.dataset import AbstractDataset, StandardPretrainDataset, StandardForecastingFinetuneDataset
from torch.utils.data import DataLoader


class WSDMDataset(AbstractDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self._dataset = None
        self.scaler = None
        self.encoder = None

    def prepare(self):
        dataset = pk.load(open("data_dir/wsdm.pk", "rb"))
        self.scaler = dataset["scaler"]
        self.encoder = dataset["encoder"]
        train, train_target = zip(*[(entry.loc[:, ["x", "y", "z"]].values, entry["activity"]) for entry in dataset["train"]])
        validation, validation_target = zip(*[(entry.loc[:, ["x", "y", "z"]].values, entry["activity"]) for entry in dataset["validation"]])
        test, test_target = zip(*[(entry.loc[:, ["x", "y", "z"]].values, entry["activity"]) for entry in dataset["test"]])

        return (train, train_target), (validation, validation_target), (test, test_target)

    def get_pretrain_loader(self):
        (train, train_target), (validation, validation_target), (test, test_target) = self.prepare()

        train_loader = DataLoader(StandardPretrainDataset(train, train, self.config), batch_size=self.config["batch_size"],
                                  num_workers=2 if not self.config["dry_run"] else 0, shuffle=self.config["shuffle"], drop_last=True,
                                  persistent_workers=not self.config["dry_run"])
        validation_loader = DataLoader(StandardPretrainDataset(validation, validation, self.config), batch_size=self.config["batch_size"],
                                       num_workers=2 if not self.config["dry_run"] else 0, shuffle=False, drop_last=True,
                                       persistent_workers=not self.config["dry_run"])
        test_loader = DataLoader(StandardPretrainDataset(test, test, self.config), batch_size=self.config["batch_size"],
                                 num_workers=2 if not self.config["dry_run"] else 0, shuffle=False, drop_last=True,
                                 persistent_workers=not self.config["dry_run"])

        return train_loader, validation_loader, test_loader


    def get_finetune_loader(self):
        (train, train_target), (validation, validation_target), (test, test_target) = self.prepare()

        train_loader = DataLoader(StandardForecastingFinetuneDataset(train, train_target, self.config),
                                  batch_size=self.config["batch_size"],
                                  num_workers=2 if not self.config["dry_run"] else 0, shuffle=self.config["shuffle"],
                                  drop_last=True, persistent_workers=not self.config["dry_run"])
        validation_loader = DataLoader(StandardForecastingFinetuneDataset(validation, validation_target,
                                                                     self.config),
                                       batch_size=self.config["batch_size"],
                                       num_workers=2 if not self.config["dry_run"] else 0, shuffle=False,
                                       drop_last=True, persistent_workers=not self.config["dry_run"])
        test_loader = DataLoader(StandardForecastingFinetuneDataset(test, test_target, self.config),
                                 batch_size=self.config["batch_size"],
                                 num_workers=2 if not self.config["dry_run"] else 0, shuffle=False, drop_last=True, persistent_workers=not self.config["dry_run"])

        return train_loader, validation_loader, test_loader

    def cv_split(self):
        yield self.get_finetune_loader()
