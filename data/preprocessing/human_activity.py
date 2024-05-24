import pandas as pd
from datetime import datetime, timedelta
import pickle as pk
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def prepare_wsdm(path="data_dir/WISDM_ar_v1.1_raw.txt"):
    df = pd.read_csv(path, header=None, names=["user", "activity", "timestamp", "x", "y", "z"], on_bad_lines="skip",
                     index_col=False)
    df["z"] = df["z"].apply(lambda x: x if isinstance(x, float) else x.replace(";", "")).astype(float)
    df["x"] = df["x"].apply(lambda x: x if isinstance(x, float) else x.replace(";", "")).astype(float)
    df["y"] = df["y"].apply(lambda x: x if isinstance(x, float) else x.replace(";", "")).astype(float)

    df.drop(columns=["user"], inplace=True)
    result = []
    for activity in df["activity"].unique():
        d = df[df["activity"] == activity][:48395]
        result.append(d)
    df = pd.concat(result)

    encoder = LabelEncoder()
    df['activity'] = encoder.fit_transform(df['activity'])

    scaler = StandardScaler()
    df.loc[:, ['x', 'y', 'z']] = scaler.fit_transform(df.loc[:, ['x', 'y', 'z']].values)
    df.loc[:, ['x', 'y', 'z']] = df.loc[:, ['x', 'y', 'z']].add(df.loc[:, ['x', 'y', 'z']].min() * -1)

    slices = split_into_slices(df, 100, .5)
    train_val, test = train_test_split(slices, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)
    pk.dump({"train": train, "validation": val, "test": test, "scaler": scaler, "encoder": encoder}, open(f"data_dir/datasets/wsdm_100_50_{datetime.now().strftime('%Y%m%d')}.pk", "wb"))

def split_into_slices(dataset, sice, data_slice: float):
    result = []
    i = 0
    count = 0
    while True:
        ds = dataset.iloc[i:i + sice, :]
        if ds.shape[0] != sice:
            break
        if max(ds["activity"].value_counts(normalize=True).to_list()) == 1:
            result.append(ds)
        else:
            count += 1

        i += int(sice * data_slice)
    print(count)
    return result


def load(path="data_dir/human_activity/ConfLongDemo_JSI.txt"):
    df = pd.read_csv(path, header=None, names=["name", "tag", "timestamp", "date", "x", "y", "z", "activity"])
    df.date = df.date.apply(lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S:%f"))

    df = df.drop(columns=["timestamp"])

    tmp = df.loc[:, ["x", "y", "z"]]
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    df.loc[:, ["x", "y", "z"]] = tmp

    results = []

    act_mapping = set()

    for seq_name, seq_values in df.groupby(df.name):

        for act, act_values in seq_values.groupby(seq_values.activity):

            act_mapping.add(act)

            act_values = act_values.assign(diff_pos=act_values.date.diff().gt(timedelta(seconds=1)).cumsum())

            for _, values in act_values.groupby(act_values.diff_pos):

                vs = []
                for tag, v in values.groupby(values.tag):
                    vs.append(v.loc[:, ["date", "x", "y", "z"]].set_index("date").add_suffix(f"_{tag}"))

                if len(vs) > 0:
                    vs = pd.concat(vs, axis=1).interpolate().dropna()
                    if vs.shape[0] >= 40:
                        results.append((act, vs))

    dataset = []
    act_code = {act: num for num, act in enumerate(act_mapping)}
    coding = [0] * (len(act_code) - 1)
    act_encoding = {act: coding[0:act_code[act]] + [1] + coding[act_code[act]:] for act in act_mapping}
    for activity, seq in results:
        subset = []
        for i in range(0, seq.shape[0], 10):
            s = seq.iloc[i:i + 40]
            if s.shape[0] == 40:
                subset.append((s, {"activity": activity,
                                   "code": act_code[activity],
                                   "activity_encoding": act_encoding[activity]}))
        dataset.append(subset)

    pk.dump(dataset, open("data_dir/human_activity/dataset.pk", "wb"))


if __name__ == '__main__':
    prepare_wsdm()
