import torch

loss_dict = {
    "mse": torch.nn.MSELoss(),
    "l1": torch.nn.L1Loss(),
    "cross_entropy": torch.nn.CrossEntropyLoss(),
    "nll": torch.nn.NLLLoss,
    "mse+mae": lambda p, t: torch.nn.MSELoss()(p, t) + torch.nn.L1Loss()(p, t)
}


class ImputationLoss:

    def __init__(self, config):
        self.config = config
        self.loss_function = loss_dict[config["loss_function_imputation"]]

    def __call__(self, prediction, target, mask):
        if prediction.shape[0] == 0:
            return 0

        if self.config["loss_imputation_mode"] == "all":
            return self.loss_function(prediction, target)

        elif self.config["loss_imputation_mode"] == "imputation":
            return self.loss_function(torch.masked_select(prediction, mask),
                                      torch.masked_select(target, mask))

        elif self.config["loss_imputation_mode"] == "weighted_imputation":
            loss_masked = self.loss_function(torch.masked_select(prediction, mask),
                                             torch.masked_select(target, mask))

            loss_unmasked = self.loss_function(torch.masked_select(prediction, ~mask),
                                               torch.masked_select(target, ~mask))
            return loss_unmasked + (loss_masked * self.config["loss_weight_alpha"])
        elif self.config["loss_imputation_mode"] == "imputation_only":
            return self.loss_function(torch.masked_select(prediction, mask),
                                      torch.masked_select(target, mask))
        else:
            raise IOError("Unknown loss mode")


class ForecastingLoss:

    def __init__(self, config):
        self.config = config
        self.loss_function = loss_dict[config["loss_function_forecasting"]]

    def __call__(self, prediction, target):
        if prediction.shape[0] == 0:
            return 0

        return self.loss_function(prediction, target)


class BinaryClassLoss:

    def __init__(self, config, use_fraction_weight=True):
        self.config = config
        self.user_fraction_weigth = use_fraction_weight

    def __call__(self, prediction, target):
        return torch.nn.functional.binary_cross_entropy(prediction, target,
                                                        weight=torch.tensor([self.config["noise_count_fraction"]],
                                                                            device=prediction.device)
                                                        if self.user_fraction_weigth else None)


class PreTrainingLoss:

    def __init__(self, config):
        self.config = config
        self.imputation_loss = ImputationLoss(config)
        self.binary_loss = BinaryClassLoss(config)

    def __call__(self, prediction_imputation, target_imputation,
                 mask,
                 prediction_binary=None, target_binary=None):
        loss = self.imputation_loss(prediction_imputation[target_binary],
                                    target_imputation[target_binary], mask[target_binary]) * \
               self.config["loss_weight_beta"]
        if prediction_binary is not None and target_binary is not None:
            loss += self.binary_loss(prediction_binary, target_binary.float()) * self.config["loss_weight_gamma"]

        return loss
