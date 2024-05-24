import pickle as pk
from data.datasets.dataset import AbstractDataset, StandardPretrainDataset, StandardImputationFinetuneDataset
from torch.utils.data import DataLoader


class AirQualityImputationSpecificSplit(AbstractDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.mask_train = None
        self.mask_val = None
        self.mask_test = None
        self._dataset = None
        self.scaler = None

    def load_data(self):
        dataset = pk.load(open("data_dir/air_quality.pk", "rb"))
        train_data = dataset["train"]
        validation_data = dataset["validation"]
        test_data = dataset["test"]
        self.scaler = dataset["scaler"]
        return train_data, validation_data, test_data

    def get_pretrain_loader(self):
        train_data, validation_data, test_data = self.load_data()


        train_loader = DataLoader(StandardPretrainDataset(train_data, train_data, self.config), batch_size=self.config["batch_size"],
                                  num_workers=2 if not self.config["dry_run"] else 0, shuffle=self.config["shuffle"], drop_last=True,
                                  persistent_workers=not self.config["dry_run"])
        validation_loader = DataLoader(StandardPretrainDataset(validation_data, validation_data, self.config), batch_size=self.config["batch_size"],
                                       num_workers=2 if not self.config["dry_run"] else 0, shuffle=False, drop_last=True,
                                       persistent_workers=not self.config["dry_run"])
        test_loader = DataLoader(StandardPretrainDataset(test_data, test_data, self.config), batch_size=self.config["batch_size"],
                                 num_workers=2 if not self.config["dry_run"] else 0, shuffle=False, drop_last=True,
                                 persistent_workers=not self.config["dry_run"])

        return train_loader, validation_loader, test_loader


    def get_finetune_loader(self):
        train_data, validation_data, test_data = self.load_data()

        train_loader = DataLoader(StandardImputationFinetuneDataset(train_data, train_data, self.config), batch_size=self.config["batch_size"],
                                  num_workers=2 if not self.config["dry_run"] else 0, shuffle=self.config["shuffle"],
                                  drop_last=True, persistent_workers=not self.config["dry_run"])
        validation_loader = DataLoader(StandardImputationFinetuneDataset(validation_data, validation_data, self.config),
                                       batch_size=self.config["batch_size"],
                                       num_workers=2 if not self.config["dry_run"] else 0, shuffle=False,
                                       drop_last=True, persistent_workers=not self.config["dry_run"])
        test_loader = DataLoader(StandardImputationFinetuneDataset(test_data, test_data, self.config), batch_size=self.config["batch_size"],
                                 num_workers=2 if not self.config["dry_run"] else 0, shuffle=False, drop_last=True, persistent_workers=not self.config["dry_run"])

        return train_loader, validation_loader, test_loader

    def cv_split(self):
        yield self.get_finetune_loader()
