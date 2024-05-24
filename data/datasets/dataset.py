import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class StandardPretrainDataset(Dataset):

    def calc_time_embedding(self, df: pd.DataFrame):
        df['date'] = pd.to_datetime(df.index)
        df['month'] = df['date'].apply(lambda row: row.month, 1)
        df['day'] = df['date'].apply(lambda row: row.day, 1)
        df['weekday'] = df['date'].apply(lambda row: row.weekday(), 1)
        df['hour'] = df['date'].apply(lambda row: row.hour, 1)
        df['minute'] = df['date'].apply(lambda row: row.minute, 1)
        df['minute'] = df["minute"].map(lambda x: x // 15)
        return df.drop(
            columns=[col for col in df.columns if col not in ["month", "day", "weekday", "hour", "minute"]]).values

    def __init__(self, data, target, config: dict):
        super().__init__()
        self.meta = None

        if isinstance(data[0], tuple):
            data, meta = zip(*data)
            target, meta = zip(*target)
            self.meta = meta

        if isinstance(data[0], pd.DataFrame):
            self._data = [entry.values for entry in data]
            self._target = [entry.values for entry in target]
        else:
            self._data = data
            self._target = target
        self._config = config

        self.time_embeddings = None
        if config.get("add_embedding", False):
            self.time_embeddings = [self.calc_time_embedding(entry) for entry in data]

    def __getitem__(self, item):

        if self.time_embeddings is None:
            return (self._data[item].astype("float32"),
                    self._target[item].astype("float32"),
                    self.calc_mask(self._data[item]))
        return (self._data[item].astype("float32"),
                self._target[item].astype("float32"),
                self.calc_mask(self._data[item]),
                self.time_embeddings[item])

    def __len__(self):
        return len(self._data)

    def calc_mask(self, data):
        mask = np.zeros_like(data)
        mask[np.where(data == -1)] = -1  # missing values are marked as -1
        mask = self.add_sequences(matrix=mask, n=self._config["mask_count"],
                                  sequence_length_range=(self._config["mask_size"],
                                                         self._config["mask_size"] + self._config["mask_var_length"]),
                                  start_offset=self._config["mask_offset"],
                                  end_offset=self._config.get("horizon", None) or self._config["mask_offset"])

        return mask

    def add_sequences(self, matrix, n, sequence_length_range=(2, 4), start_offset=2, end_offset=1):
        rows, cols = matrix.shape

        for _ in range(n):
            for col in range(cols):
                # Randomly choose the sequence length (between 2 and 3)
                sequence_length = np.random.randint(*sequence_length_range)

                # Randomly choose the starting point for each column, considering the offset
                start_row = np.random.randint(start_offset, rows - sequence_length - end_offset + 1)

                # Check if the chosen area is available
                while np.sum(matrix[start_row:start_row + sequence_length, col]) > 0:
                    start_row = np.random.randint(start_offset, rows - sequence_length - end_offset + 1)

                # Add the sequence to the matrix
                matrix[start_row:start_row + sequence_length, col] = 1  # artificial missing values are marked as 1

        return matrix


class StandardImputationFinetuneDataset(StandardPretrainDataset):

    def calc_mask(self, data):
        mask = np.zeros_like(data)
        mask[np.where(data == -1)] = -1  # missing values are marked as -1
        mask = self.apply_rm(mask, missing_ratio=self._config["missing_ratio"])
        return mask

    def apply_rm(self, mask, missing_ratio=.1):
        shape = mask.shape
        reshaped = mask.reshape(-1)
        try:
            choices = np.random.choice(
                np.where(reshaped != -1)[0].tolist(),
                size=int(reshaped.shape[0] * missing_ratio), replace=False)
            reshaped[choices] = 1  # artificial missing values are marked as 1

        except Exception as e:
            print(e)
        return reshaped.reshape(shape)


class StandardForecastingFinetuneDataset(StandardPretrainDataset):

    def __getitem__(self, item):
        if self.time_embeddings is None:
            return (self._data[item].astype("float32"),
                    self._target[item].astype("float32"))
        return (self._data[item].astype("float32"),
                self._target[item].astype("float32"),
                0.,
                self.time_embeddings[item])


class AbstractDataset:

    def __init__(self, config: dict, **kwargs):
        self.config = config

    def __len__(self): ...

    def __getitem__(self, item): ...

    def get_data(self, **kwargs): ...  # returns data, target

    def cv_split(self): ...
