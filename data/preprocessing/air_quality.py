import os
import pandas as pd
from datetime import datetime
import pickle as pk
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data_path = "data_dir/airquality/"

def load(root_path=data_path + "raw/"):
    for file in os.listdir(root_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(root_path, file))
            df.loc[:, "timestamp"] = df.apply(lambda x: datetime(year=x.loc["year"],
                                                                 month=x.loc["month"],
                                                                 day=x.loc["day"],
                                                                 hour=x.loc["hour"]), axis=1)
            df = df.set_index("timestamp")
            df = df.drop(columns=["No", "year", "month", "day", "hour", "wd", "station"])

            df = df.reset_index(drop=False).rename(columns={"index": "timestamp"})

            df = df.assign(month=df.timestamp.apply(lambda x: x.month))

            yield df

def chunk_data(data, data_window_size=24):
    dataset = []
    for chunk in data:
        dataset += [chunk.iloc[i * data_window_size:(i + 1) * data_window_size].values
                    for i in range(chunk.shape[0] // data_window_size)
                    if chunk.iloc[i * data_window_size:(i + 1) * data_window_size].shape[0] == data_window_size]
    return dataset

def scale_and_shape(data, scaler):
    data = pd.DataFrame(scaler.transform(data))
    data = data.subtract(data.min())
    assert data.min().max() == 0
    data = data.fillna(-1)
    return data

def prepare_pretraining_imputation(data_window_size=24, shuffle=False):
    train, val, test = [], [], []
    dfs = []
    scaler = StandardScaler()
    for df in load():
        dfs.append(df)
        scaler = scaler.partial_fit(df.drop(columns=["timestamp", "month"]).values)

    for df in dfs:
        unique_months = df["timestamp"].dt.to_period("M").unique()
        test_set = df[df["timestamp"].dt.to_period("M").isin(unique_months[:10])].drop(columns=["timestamp", "month"])
        val_set = df[df["timestamp"].dt.to_period("M").isin(unique_months[10:20])].drop(columns=["timestamp", "month"])
        train_set = df[df["timestamp"].dt.to_period("M").isin(unique_months[20:])].drop(columns=["timestamp", "month"])
        train.append(scale_and_shape(train_set, scaler))
        val.append(scale_and_shape(val_set, scaler))
        test.append(scale_and_shape(test_set, scaler))

    train, val, test = chunk_data(train, data_window_size), chunk_data(val, data_window_size), chunk_data(test, data_window_size)

    data = {"train": train,
            "validation": val,
            "test": test,
            "scaler": scaler}
    pk.dump(data,
            open(f"data_dir/datasets/air_quality_pretrain_imputation_standard_scaled_shuffle={shuffle}_fix_split_{datetime.now().strftime('%Y%m%d')}.pk", "wb"))


if __name__ == "__main__":
    prepare_pretraining_imputation()
