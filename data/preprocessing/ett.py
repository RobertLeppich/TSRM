import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle as pk


def prepare(root_path="data_dir/ETTsmall/ETTm1.csv"):
    data = pd.read_csv(root_path).set_index("date")
    scaler = StandardScaler()
    data.iloc[:, :] = scaler.fit_transform(data.values)
    data = data.add(data.min() * -1)
    pk.dump({"dataset": data, "scaler": scaler}, open("data_dir/datasets/ett_small_m1_z-transformed.pk", "wb"))


if __name__ == '__main__':
    prepare()
