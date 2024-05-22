from torchmetrics.functional.regression import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from torchmetrics.functional.classification import f1_score, precision, recall
import torch
import numpy as np

def rescale(data, scaler, meta):
    result = []
    for i in range(len(data)):
        batch = data[i]
        scale = scaler[meta[i]] if meta is not None else scaler
        result.append(scale.inverse_transform(batch.view(-1, batch.shape[1])).reshape(batch.shape))
    return torch.tensor(np.stack(result))

def calc_metrics(output=None, target=None, mask=None, scaler=None, classification_output=None, classification_target=None, meta=None,
                 calc_real=False, prefix="", classification_finetune=False, num_classes=None):
    if mask is None:
        mask = np.zeros_like(output)


    if classification_finetune:
        classification_loss = {
            # classification
            "f1_classification": float(f1_score(torch.argmax(classification_output, -1), classification_target, task="multiclass", num_classes=num_classes)),
            "precision_classification": float(precision(torch.argmax(classification_output, -1), classification_target, task="multiclass", num_classes=num_classes)),
            "recall_classification": float(recall(torch.argmax(classification_output, -1), classification_target, task="multiclass", num_classes=num_classes))
        }
        return classification_loss

    elif classification_output is not None:
        classification_loss = {
            # classification
            "f1_classification": float(f1_score(classification_output, classification_target, task="binary")),
            "precision_classification": float(precision(classification_output, classification_target, task="binary")),
            "recall_classification": float(recall(classification_output, classification_target, task="binary"))
        }
    else:
        classification_loss = {}

    ts_loss = {
        # all
        prefix + "mae_all": float(mean_absolute_error(output, target)),
        prefix + "mse_all": float(mean_squared_error(output.reshape(-1), target.reshape(-1))),
        prefix + "rmse_all": float(torch.sqrt(mean_squared_error(output.reshape(-1), target.reshape(-1)))),

        # missing
        prefix + "mae_missing": float(mean_absolute_error(output[mask], target[mask])),
        prefix + "mse_missing": float(mean_squared_error(output[mask].reshape(-1), target[mask].reshape(-1))),
        prefix + "rmse_missing": float(torch.sqrt(mean_squared_error(output[mask].reshape(-1), target[mask].reshape(-1)))),

        # reproduce
        prefix + "mae_reproduce": float(mean_absolute_error(output[~mask], target[~mask])),
        prefix + "mse_reproduce": float(mean_squared_error(output[~mask].reshape(-1), target[~mask].reshape(-1))),
        prefix + "rmse_reproduce": float(torch.sqrt(mean_squared_error(output[~mask].reshape(-1), target[~mask].reshape(-1)))),

    }

    real_loss = {}
    if calc_real:
        rescaled_target = rescale(target.detach().cpu(), scaler, meta)
        rescaled_output = rescale(output.detach().cpu(), scaler, meta)
        mask = mask.detach().cpu()
        real_loss = {
            # all real
            "mae_all_real": float(mean_absolute_error(rescaled_output, rescaled_target)),
            "mse_all_real": float(mean_squared_error(rescaled_output.reshape(-1), rescaled_target.reshape(-1))),
            "rmse_all_real": float(torch.sqrt(mean_squared_error(rescaled_output.reshape(-1), rescaled_target.reshape(-1)))),

            # missing real
            "mae_missing_real": float(mean_absolute_error(rescaled_output[mask], rescaled_target[mask])),
            "mse_missing_real": float(mean_squared_error(rescaled_output[mask].reshape(-1), rescaled_target[mask].reshape(-1))),
            "rmse_missing_real": float(torch.sqrt(mean_squared_error(rescaled_output[mask].reshape(-1), rescaled_target[mask].reshape(-1)))),

            # reproduce real
            "mae_reproduce_real": float(mean_absolute_error(rescaled_output[~mask], rescaled_target[~mask])),
            "mse_reproduce_real": float(mean_squared_error(rescaled_output[~mask].reshape(-1), rescaled_target[~mask].reshape(-1))),
            "rmse_reproduce_real": float(torch.sqrt(mean_squared_error(rescaled_output[~mask].reshape(-1), rescaled_target[~mask].reshape(-1)))),

        }

    return {**ts_loss, **classification_loss, **real_loss}
