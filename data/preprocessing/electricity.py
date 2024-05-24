import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle as pk
from datetime import datetime
from datetime import date


def clean_data_standard_scaling(df):
    df.index = pd.to_datetime(df.index)
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df))
    scaled_df.index = df.index
    scaled_df.columns = df.columns

    result = dict()
    for col in scaled_df.columns:
        col_values = scaled_df.loc[:, col]
        col_values = col_values.loc[col_values.ne(0).idxmax():]

        # ensure min values is 0
        col_values = col_values.subtract(col_values.min())
        assert col_values.min() == 0

        result[col] = pd.DataFrame(col_values)

    return result, scaler

def prepare_strict_split(shuffle=True, validation_size=.2, test_size=.3):
    df = pd.read_csv("data_dir/electricity_data/LD2011_2014.txt", sep=";", index_col=0, decimal=",")

    data, scaler = clean_data_standard_scaling(df)

    def split(chunk):
        ds = chunk.assign(date=[x.date() for x in chunk.index])
        test = ds.loc[(ds.date >= date(2011, 1, 1)) & (ds.date <= date(2011, 10, 31))]
        validation = ds.loc[(ds.date >= date(2011, 11, 1)) & (ds.date <= date(2012, 8, 31))]
        train = ds.loc[(ds.date >= date(2012, 9, 1)) & (ds.date <= date(2014, 12, 31))]
        return train.drop(columns=["date"]), validation.drop(columns=["date"]), test.drop(columns=["date"])

    train_result, validation_result, test_result = dict(), dict(), dict()
    for col, col_values in data.items():
        train_result[col], validation_result[col], test_result[col] = split(col_values)

    data = {"train": train_result,
            "validation": validation_result,
            "test": test_result,
            "scaler": scaler}

    pk.dump(data, open(
        f"data_dir/datasets/electricity_standard_scaled_strict_split_shuffle={shuffle}_validation={validation_size}"
        f"_test={test_size}_{datetime.now().strftime('%Y%m%d')}.pk",
        "wb"))


if __name__ == '__main__':
    prepare_strict_split()
