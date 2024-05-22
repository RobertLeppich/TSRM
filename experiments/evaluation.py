from database import LS2MongoDB
import os
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


config_of_interest = ["N", "h", "encoding_size", "attention_func", "conv_dims", "missing_ratio", "add_pooled_representations", "shuffle_seed"]
metric_of_interest = ["loss", "mse_missing", "mae_missing", "96_mse_missing", "96_mae_missing", "mse_reproduce", "mae_reproduce", "mse_all", "mae_all", "f1_classification"]

def load_best_runs(run_id, metric="loss"):
    coll = LS2MongoDB().get_database("MA_Paper").get_collection("experiment_runs")
    entry = coll.find_one({"run_id": run_id})
    pretrain_runs = sorted(entry["pretrain_runs"], key=lambda x: x["metrics"].get(metric, 100))
    finetune_runs = sorted(entry["finetune_runs"], key=lambda x: x["metrics"].get(metric, 100))
    return pretrain_runs, finetune_runs

def load_run_ids(exp_id: str, codename: str = None):
    coll = LS2MongoDB().get_database("MA_Paper").get_collection("experiment_runs")
    query = {"exp_id": exp_id}
    if codename is not None:
        query.update({"codename": codename})
    return [(entry["run_id"], entry.get("codename", "N/A")) for entry in coll.find(query)]

def _extract_config(entry):
    return [f"{k}: {entry['config'].get(k, None)}" for k in config_of_interest]


def correlation(runs, metrics):
    result = []
    for run, codename in runs:
        result.append({"codename": codename, "para_run_id": run["parameter_run_id"], **{k: run["metrics"].get(k, float("inf")) for k in metric_of_interest}, **{k: run["config"].get(k, None) for k in config_of_interest}})

    df = pd.DataFrame(result).sort_values(metrics)
    df = df.assign(rank=df.index)

    sns.heatmap(df.drop(columns=["codename", "conv_dims", "attention_func", "para_run_id"]).corr())
    build_latex_table(df, [0, 1, 2, 14, 3, 10, 16])
    plt.tight_layout()
    plt.show()
    pass

def evaluate_experiment(exp_id, codename=None, metric="loss"):
    best_runs_pretrain = []
    best_runs_finetune = []
    for run_id, exp_codename in load_run_ids(exp_id, codename):
        pretrain, finetune = load_best_runs(run_id, metric=metric)
        best_runs_pretrain += [(entry, exp_codename) for entry in pretrain]
        best_runs_finetune += [(entry, exp_codename) for entry in finetune]

    best_runs_pretrain = sorted(best_runs_pretrain, key=lambda x: x[0]["metrics"][metric])
    best_runs_finetune = sorted(best_runs_finetune, key=lambda x: x[0]["metrics"].get(metric, float("inf")))

    #result_pretrain = [f"{entry['metrics'].get('loss', 100)}: {', '.join(_extract_config(entry))}" for entry in best_runs_pretrain]
    #result_finetune = [f"{entry['metrics'].get('loss', 100)}: {', '.join(_extract_config(entry))}" for entry in best_runs_finetune]
    correlation(best_runs_pretrain, metric)
    correlation(best_runs_finetune, metric)


def build_latex_table(df, rows=[]):
    result = []
    def format(entry, col):
        if col == "conv_dims":
            conv_res = []
            for cov in entry:
                conv_res.append(f"k: {cov[0]*100:.0f}, d: {cov[1]}")
            r = r" \\ ".join(conv_res)
            return r" \begin{tabular}[c]{@{}l@{}}" + r + r" \end{tabular} &"
        if isinstance(entry, float):
            return np.round(entry, 3)
        else:
            return entry

    cols = ["N", "h", "encoding_size", "attention_func", "conv_dims", "mse_reproduce", "mse_missing", "f1_classification"]
    r_c = 0
    for idx, values in df.iterrows():
        if idx not in rows:
            continue
        row_str = r"\rowcolor[HTML]{C0C0C0} " if r_c % 2 == 0 else ""
        row_str += " & ".join([f"{format(values[c], c)}" for c in cols]) + r" \\ "
        result.append(row_str)
        r_c += 1
    result = "".join(result)
    print(result)

if __name__ == '__main__':
    evaluate_experiment(exp_id="Exp4", metric="mse_missing", codename="52. run")