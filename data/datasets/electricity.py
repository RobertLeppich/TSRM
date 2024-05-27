import pickle as pk
from data.datasets.dataset import StandardPretrainDataset, StandardImputationFinetuneDataset, StandardForecastingFinetuneDataset
from torch.utils.data import DataLoader


class ElectricityImputationSpecificSplitUnivariate:
    def __init__(self, config, **kwargs):
        self.config = config

    def load_data(self):

        dataset = pk.load(open("data_dir/electricity.pk", "rb"))
        train_data = dataset["train"]
        validation_data = dataset["validation"]
        test_data = dataset["test"]
        self.scaler = dataset["scaler"]

        train_data = self.split_into_slices(train_data, self.config["data_window_size"],
                                            self.config.get("data_slice", 1.))
        validation_data = self.split_into_slices(validation_data, self.config["data_window_size"], 1)
        test_data = self.split_into_slices(test_data, self.config["data_window_size"], 1)

        return train_data, validation_data, test_data

    def split_into_slices(self, dataset_dict, sice, data_slice: float):
        result = []
        for client, dataset in dataset_dict.items():
            i = 0

            while True:
                ds = dataset.iloc[i:i + sice, :].values
                if ds.shape[0] != sice:
                    break
                result.append(ds)
                i += int(sice * data_slice)
        return result

    def get_pretrain_loader(self):
        train_data, validation_data, test_data = self.load_data()

        train_loader = DataLoader(StandardPretrainDataset(train_data, train_data, self.config),
                                  batch_size=self.config["batch_size"],
                                  num_workers=4 if not self.config["dry_run"] else 0, shuffle=self.config["shuffle"],
                                  drop_last=True,
                                  persistent_workers=not self.config["dry_run"])
        validation_loader = DataLoader(StandardPretrainDataset(validation_data, validation_data, self.config),
                                       batch_size=self.config["batch_size"],
                                       num_workers=4 if not self.config["dry_run"] else 0, shuffle=False,
                                       drop_last=True,
                                       persistent_workers=not self.config["dry_run"])
        test_loader = DataLoader(StandardPretrainDataset(test_data, test_data, self.config),
                                 batch_size=self.config["batch_size"],
                                 num_workers=4 if not self.config["dry_run"] else 0, shuffle=False, drop_last=True,
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


class ElectricityForecastingSpecificSplitUnivariat(ElectricityImputationSpecificSplitUnivariate):

    def get_finetune_loader(self):
        train_data, validation_data, test_data = self.load_data()

        input_size, horizon = self.config["data_window_size"] - self.config["horizon"], self.config["horizon"]

        train_loader = DataLoader(
            StandardForecastingFinetuneDataset([entry for entry in self.mask_horizon(train_data, horizon)],
                                              [entry[-horizon:, [-1]] for entry in train_data], self.config),
            batch_size=self.config["batch_size"],
            num_workers=2 if not self.config["dry_run"] else 0, shuffle=self.config["shuffle"],
            drop_last=True, persistent_workers=not self.config["dry_run"])
        validation_loader = DataLoader(
            StandardForecastingFinetuneDataset([entry for entry in self.mask_horizon(validation_data, horizon)],
                                              [entry[-horizon:, [-1]] for entry in validation_data],
                                              self.config),
            batch_size=self.config["batch_size"],
            num_workers=2 if not self.config["dry_run"] else 0, shuffle=False,
            drop_last=True, persistent_workers=not self.config["dry_run"])
        test_loader = DataLoader(
            StandardForecastingFinetuneDataset([entry for entry in self.mask_horizon(test_data, horizon)],
                                              [entry[-horizon:, [-1]] for entry in test_data], self.config),
            batch_size=self.config["batch_size"],
            num_workers=2 if not self.config["dry_run"] else 0, shuffle=False, drop_last=True,
            persistent_workers=not self.config["dry_run"])

        return train_loader, validation_loader, test_loader

    def mask_horizon(self, dataset, horizon):
        result = []
        for ds in dataset:
            ds = ds.copy()
            ds[-horizon:, :] = -1
            result.append(ds)
        return result