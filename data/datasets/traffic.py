from data.datasets.dataset import AbstractDataset, StandardPretrainDataset, StandardForecastingFinetuneDataset
from torch.utils.data import DataLoader
import pandas as pd


def transform_univariate(data):
    result = []
    for entry in data:
        for i in range(entry.shape[1]):
            e = entry[:, i]
            if e.max() == 0 or sum(e) < 0:
                continue
            result.append(e.reshape(-1, 1))
    return result


class TrafficDataset(AbstractDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self._dataset = None
        self.scaler = None

    def split_into_slices(self, dataset, sice, data_slice: float, univariate):
        result = []
        i = 0
        while True:
            ds = dataset.iloc[i:i+sice, :]
            if ds.shape[0] != sice:
                break
            if not univariate:
                result.append(ds)
            else:
                result += self.transfer_into_univariate(ds)
            i += int(sice*data_slice)
        return result

    def transfer_into_univariate(self, data):
        result = []
        for col in data.columns:
            result.append(data.loc[:, [col]])
        return result

    def prepare(self):
        ds = pd.read_csv("data_dir/traffic.csv").set_index("date")

        train_size, validation_size, test_size = self.config["validation_size"] + self.config["test_size"], self.config["validation_size"], self.config["test_size"]
        train_size, validation_size, test_size = int(train_size*len(ds)), int(validation_size*len(ds)), int(test_size*len(ds))

        train_data = ds.iloc[:train_size]
        validation_data = ds.iloc[train_size:train_size+validation_size]
        test_data = ds.iloc[train_size+validation_size:]

        train_data = self.split_into_slices(train_data, self.config["data_window_size"], self.config.get("data_slice", 1.), univariate=True)
        validation_data = self.split_into_slices(validation_data, self.config["data_window_size"], 1, univariate=True)
        test_data = self.split_into_slices(test_data, self.config["data_window_size"],1, univariate=True)
        return train_data, validation_data, test_data


    def get_pretrain_loader(self):

        train_data, validation_data, test_data = self.prepare()

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
        train_data, validation_data, test_data = self.prepare()

        input_size, horizon = self.config["data_window_size"] - self.config["horizon"], self.config["horizon"]

        train_loader = DataLoader(StandardForecastingFinetuneDataset([entry for entry in self.mask_horizon(train_data, horizon)],
                                                                     [entry.iloc[-horizon:, [-1]] for entry in train_data],
                                                                     self.config),
                                  batch_size=self.config["batch_size"],
                                  num_workers=2 if not self.config["dry_run"] else 0, shuffle=self.config["shuffle"],
                                  drop_last=True, persistent_workers=not self.config["dry_run"])
        validation_loader = DataLoader(StandardForecastingFinetuneDataset([entry for entry in self.mask_horizon(validation_data, horizon)],
                                                                     [entry.iloc[-horizon:, [-1]] for entry in validation_data],
                                                                     self.config),
                                       batch_size=self.config["batch_size"],
                                       num_workers=2 if not self.config["dry_run"] else 0, shuffle=False,
                                       drop_last=True, persistent_workers=not self.config["dry_run"])
        test_loader = DataLoader(StandardForecastingFinetuneDataset([entry for entry in self.mask_horizon(test_data, horizon)],
                                                                     [entry.iloc[-horizon:, [-1]] for entry in test_data],
                                                                     self.config),
                                 batch_size=self.config["batch_size"],
                                 num_workers=2 if not self.config["dry_run"] else 0, shuffle=False, drop_last=True, persistent_workers=not self.config["dry_run"])

        return train_loader, validation_loader, test_loader

    def mask_horizon(self, dataset, horizon):
        result = []
        for ds in dataset:
            ds = ds.copy()
            ds.iloc[-horizon:, :] = -1
            result.append(ds)
        return result

    def cv_split(self):
        yield self.get_finetune_loader()
