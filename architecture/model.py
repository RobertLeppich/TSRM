from typing import Any, Literal, Mapping
import torch
from sympy.abc import lamda
from torch.cuda import device
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import lightning as pl
from architecture.utils import add_noise, build_mask, build_mask_from_data
from architecture.loss_functions import PreTrainingLoss, ImputationLoss, ForecastingLoss, BinaryClassLoss
from architecture.multiHeadAttention import MultiHeadedAttention
from architecture.metrics import calc_metrics
from architecture.RevIN import RevIN
from embedding.data_embedding import DataEmbedding
import numpy as np
from experiments.plot import compare_ts
from matplotlib import pyplot as plt
import seaborn as sns


class EncodingLayer(pl.LightningModule):

    def __init__(self,
                 h: int,
                 encoding_size: int,
                 dropout: float,
                 seq_len: int,
                 pred_len: int,
                 conv_dims: tuple,  # ((kernel_size, dilation, groups), (...))
                 attention_func: Literal["classic", "propsparse", "propsparse_entmax15"],
                 add_pooled_representations: bool = False,
                 ts_attention=True,
                 clip_data=0,
                 split_feature_encoding=False,
                 feature_dimension=1,
                 n_kernel=1,
                 calc_attention_map=False,
                 feature_ff=False,
                 **kwargs):
        super().__init__()

        seq_len = seq_len + (2 * clip_data)
        self.feature_ff = feature_ff
        self._self_attention = MultiHeadedAttention(encoding_size=encoding_size * n_kernel, h=h, dropout=dropout,
                                                    attention_func=attention_func, ts_attention=ts_attention,
                                                    split_feature_encoding=split_feature_encoding,
                                                    feature_dimension=feature_dimension,
                                                    **kwargs)
        self.feature_dimension = feature_dimension
        self.single_feature_encoding = encoding_size
        if split_feature_encoding:
            encoding_size *= feature_dimension

        self.calc_attention_map = calc_attention_map
        self._dropout = nn.Dropout(p=dropout)
        self.ts_attention = ts_attention
        self.representation_layers = nn.ModuleList()
        self.merge_layers = nn.ModuleList()
        self.pools = nn.ModuleList()

        conv_size = 0
        for kernel_s, dl, groups in conv_dims:
            if groups == -1:
                groups = encoding_size

            stride = kernel_s  # max(kernel_s // 2, 2)
            conv_size_layer = self.conv_output_dimension(seq_len, dl, 0, kernel_s, stride)
            trans_size_layer = self.transpose_output_dimension(conv_size_layer, dl, 0, kernel_s, stride)
            conv_size += conv_size_layer

            repr_conv = nn.Conv1d(in_channels=encoding_size,
                                  out_channels=encoding_size * n_kernel,
                                  kernel_size=kernel_s,
                                  stride=stride,
                                  dilation=int(dl),
                                  groups=groups)

            self.representation_layers.append(repr_conv)

            t_conv = nn.ConvTranspose1d(in_channels=encoding_size * n_kernel,
                                        out_channels=encoding_size,
                                        kernel_size=kernel_s,
                                        output_padding=seq_len - trans_size_layer,
                                        stride=stride,
                                        dilation=int(dl),
                                        groups=groups)

            self.merge_layers.append(t_conv)

            if add_pooled_representations:
                stride = 3
                kernel_s = 3

                self.pools.append(nn.MaxPool1d(kernel_size=kernel_s,
                                               stride=stride))
                conv_size += int((int(conv_size_layer - (kernel_s - 1) - 1) / stride) + 1)

        self.conv_size = conv_size

        self.fin_ff = nn.Linear(encoding_size * len(conv_dims), encoding_size)

        self._feedForward = PositionalFeedforward(encoding_size * n_kernel * (self.feature_dimension if feature_ff else 1))

        self._layerNorm1 = nn.GroupNorm(num_channels=encoding_size * n_kernel,
                                        num_groups=1)
        self._layerNorm2 = nn.GroupNorm(num_channels=encoding_size * n_kernel,
                                        num_groups=1)

    def forward(self, data: torch.Tensor, global_residual) -> (torch.Tensor, torch.Tensor):
        representations = [nn.functional.elu(repr_layer(data.transpose(-1, -2))) for repr_layer in
                           self.representation_layers]

        if len(self.pools) != 0:
            representations += [self.pools[i](representations[i]) for i in range(len(self.pools))]

        x = torch.cat(representations, dim=-1).transpose(-1, -2)
        representations = [entry.shape[2] for entry in representations]

        if global_residual is not None:
            x += global_residual
        else:
            global_residual = x.clone()

        # Self attention
        residual = x
        x = self._layerNorm1(x.transpose(-1, -2)).transpose(-1, -2)
        x = nn.functional.gelu(x)

        x, attn = self._self_attention(query=x, key=x, value=x)

        x = self._dropout(x)
        x += residual

        # Feed forward
        residual = x
        x = self._layerNorm2(x.transpose(-1, -2)).transpose(-1, -2)
        x = nn.functional.gelu(x)

        if self.feature_ff:
            pre_shape = x.shape
            x = x.reshape(-1, self.feature_dimension, self.conv_size, self.single_feature_encoding).transpose(-2, -1).flatten(1,2).transpose(1,-1)
        x = self._feedForward(x)
        if self.feature_ff:
            x = x.reshape(-1, self.conv_size, self.feature_dimension, self.single_feature_encoding).transpose(1, 2).flatten(0,1)

        x = self._dropout(x)
        x += residual

        global_residual += x

        x = torch.split(x, representations, dim=1)

        x = [self.merge_layers[i](x[i].transpose(-1, -2)).transpose(-1, -2)
             for i in range(len(self.representation_layers))]

        x = torch.cat(x, dim=-1)

        if self.calc_attention_map and attn is not None:
            reverse_attn = self.reverse_attn_dim(attn, representations, size=data.shape[1])
        else:
            reverse_attn = None

        x = self.fin_ff(x)
        return x, global_residual, reverse_attn

    def reverse_attn_dim(self, x, representations, size):
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        x = (x - mean) / (std + 1e-7)
        # x = (x - x.min()) / (x.max() - x.min())
        splits = torch.split(x, representations, dim=1)

        repr_layers = splits[:len(self.representation_layers)]

        merged = [nn.functional.conv_transpose1d(repr_layers[i].transpose(-1, -2),
                                                 torch.ones((x.shape[-1], x.shape[-1],
                                                             self.representation_layers[i].kernel_size[0]),
                                                            device=x.device),
                                                 dilation=self.representation_layers[i].dilation,
                                                 stride=self.representation_layers[i].stride,
                                                 groups=1).transpose(-1, -2)
                  for i in range(len(repr_layers))]
        merged = [entry if entry.shape[1] == size else
                  nn.functional.pad(entry.transpose(-1, -2), (size - entry.shape[1], 0)).transpose(-1, -2)
                  for entry in merged]

        merged = torch.cat(merged, dim=-1).sum(-1)
        # mean = merged.mean(dim=(1, 2), keepdim=True)
        # std = merged.std(dim=(1, 2), keepdim=True)
        # merged = (merged - mean) / (std + 1e-7)
        return merged

    def _calc_padding(self, input_size, window_size, stride, dilation, kernel_size):
        return (window_size - (input_size - 1) * stride - dilation * (kernel_size - 1) - 1) / (-2)

    @staticmethod
    def conv_output_dimension(window_size, dilation, padding, kernel_size, stride):
        return int(((window_size + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)

    @staticmethod
    def transpose_output_dimension(window_size, dilation, padding, kernel_size, stride):
        return (window_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1


class PositionalFeedforward(pl.LightningModule):

    def __init__(self, encoding_size):
        super().__init__()
        self.lin1 = nn.Linear(encoding_size, encoding_size)
        self.lin2 = nn.Linear(encoding_size, encoding_size)

    def forward(self, data):
        data = self.lin1(data)
        data = nn.functional.relu(self.lin2(data))
        return data


class Transformations(nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.global_minimums = None
        self.requires_transform = False
        self.revin = RevIN(num_features=config["feature_dimension"], affine=True, subtract_last=False) if config.get(
            "revin", True) else None

        self.batch_size = config["batch_size"]
        self.feature_dimension = config["feature_dimension"]

        if self.config["phase"] != "pretrain" and self.config["task"] == "forecasting":
            self.finetuning_fc = True
            self.pred_len = self.config["pred_len"]
        else:
            self.finetuning_fc = False
            self.pred_len = 0

    def transform(self, x, mask):
        # # # Normalization from Non-stationary Transformer
        # self.means = x.mean(1, keepdim=True).detach()
        # x = x - self.means
        # self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x /= self.stdev

        # revin
        if self.revin is not None:
            x = self.revin(x, "norm")

        # move value range above 0
        # m = x.min()
        # self.requires_transform = m < 0
        #
        # if self.requires_transform:
        #     self.global_minimums = x.min(1)
        #     x = x + torch.abs(self.global_minimums.values).unsqueeze(1)

        if mask is not None:
            x = torch.masked_fill(x, mask, -1)

        x = x.transpose(-2, -1).reshape(self.batch_size * self.feature_dimension, -1).unsqueeze(-1)
        return x

    def reverse(self, x):

        x = x.squeeze(-1).reshape(self.batch_size, self.feature_dimension, -1).transpose(-2, -1)
        # if self.requires_transform:
        #     x = x - torch.abs(self.global_minimums.values).unsqueeze(1)

        # # De-Normalization from Non-stationary Transformer
        # x = x * (self.stdev[:, 0, :].unsqueeze(1).repeat(1, self.config["seq_len"] + self.pred_len, 1))
        # x = x + (self.means[:, 0, :].unsqueeze(1).repeat(1, self.config["seq_len"] + self.pred_len, 1))

        if self.revin is not None:
            x = self.revin(x, "denorm")

        return x


class TSRM(pl.LightningModule):

    def __init__(self, config: dict, learning_rate: float = 0.001):
        super().__init__()

        self.feature_dimension = config["feature_dimension"]
        self.encoding_size = config["encoding_size"]
        self.seq_len = config["seq_len"] + (config.get("clip_data", 0) * 2)
        self.config = config
        self.h = config["h"]
        self.learning_rate = learning_rate
        self.lr = learning_rate
        self.res_dropout = nn.Dropout(config["dropout"])

        self.calc_attention_map = config.get("calc_attention_map", False)

        self.encoding_ff = DataEmbedding(config)

        self.layers_encoding = nn.ModuleList([
            EncodingLayer(**config)
            for _ in range(config["N"])
        ])

        self.fin_ff = nn.Linear(self.encoding_size * config["seq_len"],
                                config["pred_len"] if config["pred_len"] > 0 else config["seq_len"], bias=True) \
            if not config.get("split_feature_encoding", False) \
            else nn.ModuleList([nn.Linear(self.encoding_size, 1, bias=True)
                                for _ in range(config["feature_dimension"])])

        self.value_transformer = Transformations(config)

        self.float()

    def forward(self, encoding: torch.Tensor, x_mark, mask=None):
        orig_shape = encoding.shape

        encoding = self.value_transformer.transform(encoding, mask)

        encoding = self.encoding_ff(encoding, x_mark)

        reverse_attn_list = []
        residual = None

        # Encoding stack
        for i in range(len(self.layers_encoding)):
            encoding, residual, attn_map = self.layers_encoding[i](encoding, residual)
            reverse_attn_list.append(attn_map)

        if not self.config.get("split_feature_encoding", False):
            encoding = self.fin_ff(torch.flatten(encoding, -2, -1))

        else:
            splits = torch.split(encoding, self.encoding_size, dim=-1)
            splits = [self.fin_ff[i](splits[i]) for i in range(len(splits))]
            encoding = torch.cat(splits, dim=-1)

        if self.calc_attention_map and self.config["attention_func"] != "no":
            # B X N X len X f
            attn_map = torch.stack([entry.transpose(-1, -2).reshape(orig_shape) for entry in reverse_attn_list], dim=0).transpose(0, 1)
        else:
            attn_map = None

        encoding = self.value_transformer.reverse(encoding)
        return encoding, attn_map

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr or self.learning_rate,
                                     weight_decay=self.config.get("weight_decay", 0.0001))
        if (lr_optimizer_str := self.config.get("lr_optimizer", None)) is not None:
            lr_scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=2)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "loss"}}
        return optimizer

    def training_step(self, input_batch, idx):
        meta = dict()
        embedding = None
        if len(input_batch) == 2:
            input_data, input_target = input_batch
        elif len(input_batch) == 3:
            input_data, input_target, meta = input_batch
        else:
            input_data, input_target, embedding_x, embedding_y = input_batch

        loss = self._run(input_data, input_target, embedding_x, embedding_y, determine_metrics=False, phase="train")
        return loss

    def validation_step(self, input_batch, idx):

        meta = None
        embedding = None

        if len(input_batch) == 2:
            input_data, input_target = input_batch
        elif len(input_batch) == 3:
            input_data, input_target, meta = input_batch
        else:
            input_data, input_target, embedding_x, embedding_y = input_batch

        loss = self._run(input_data, input_target, embedding_x, embedding_y, determine_metrics=True, phase="val")
        return loss

    def test_step(self, input_batch, idx):
        meta = dict()
        embedding = None

        if len(input_batch) == 2:
            input_data, input_target = input_batch
        elif len(input_batch) == 3:
            input_data, input_target, meta = input_batch
        else:
            input_data, input_target, embedding_x, embedding_y = input_batch

        loss = self._run(input_data, input_target, embedding_x, embedding_y, determine_metrics=True,
                         calc_real=self.config.get("minmax_scaled", False), phase="test")
        return loss

    def _run(self, input_data, input_target, embedding_x, embedding_y, determine_metrics=True, calc_real=False,
             phase="train"):
        ...


class TSRMPretrain(TSRM):
    def __init__(self, config: dict):

        super().__init__(config)
        self.loss = PreTrainingLoss(config)

    def _run(self, iteration_batch, input_target, embedding_x, embedding_y, determine_metrics=True, calc_real=False,
             phase="train"):
        # reshape input
        try:
            input_data = iteration_batch.view(self.config["batch_size"],
                                              self.config["seq_len"],
                                              self.config["feature_dimension"]).float()
        except Exception:
            print(f"Error: {iteration_batch.shape}")
            return

        # build random mask
        mask = build_mask(*input_data.shape, data_batch=input_data, **self.config).to(self.device)

        # mask data
        input_data = torch.masked_fill(input_data, mask.pow(2).bool(), 0.)

        output, attn_map = self.forward(input_data, embedding_x, mask=mask)

        loss = self.loss(prediction_imputation=output.float(),
                         target_imputation=iteration_batch.float(),
                         mask=mask.eq(1))
        if determine_metrics:
            metrics = calc_metrics(output=output, target=iteration_batch,
                                   mask=mask.eq(1), calc_real=calc_real, prefix=f"{phase}_")
            metrics.update({"loss": float(loss)})
            metrics.update({phase + "_loss": float(loss)})

            self.log_dict(metrics, batch_size=input_data.shape[0])
            # [logger.log_metrics(metrics) for logger in self.loggers]
        return loss


class TSRMImputation(TSRM):
    def __init__(self, config: dict):

        super().__init__(config)
        self.loss = ImputationLoss(config)

    def _run(self, input_data, input_target, embedding_x, embedding_y, determine_metrics=True, calc_real=False,
             phase="train"):
        try:
            input_data = input_data.view(self.config["batch_size"],
                                         self.config["seq_len"],
                                         input_data.shape[-1]).float()
        except Exception:
            print(f"Error: {input_data.shape}")
            return

        mask = self.calc_mask(input_data)
        masked_data = torch.masked_fill(input_data, mask.pow(2).bool(), 0)

        output, attn_map = self.forward(masked_data, embedding_x)

        loss = self.loss(prediction=output,
                         target=input_data,
                         mask=mask.eq(1))
        if determine_metrics:
            metrics = calc_metrics(output=output, target=input_data,
                                   mask=mask.eq(1))
            metrics.update({"loss": float(loss)})
            self.log_dict(metrics, batch_size=input_data.shape[0])

        return loss

    def calc_mask(self, data):
        mask = torch.zeros_like(data, device=data.device)
        mask[torch.where(data.isnan())] = -1  # missing values are marked as -1
        mask = self.apply_rm(mask, missing_ratio=self.config["missing_ratio"])
        return mask

    def apply_rm(self, mask, missing_ratio=.1):
        shape = mask.shape
        reshaped = mask.reshape(-1)
        try:
            choices = torch.where(reshaped != -1)[0]
            choices = choices[torch.randperm(len(choices))][:int(len(choices) * missing_ratio)]
            reshaped[choices] = 1  # artificial missing values are marked as 1

        except Exception as e:
            print(e)
        return reshaped.reshape(shape)


class TSRMForecasting(TSRM):
    def __init__(self, config: dict):

        super().__init__(config)
        self.loss = ForecastingLoss(config)

    def _run(self, input_data, input_target, embedding_x, embedding_y, determine_metrics=True, calc_real=False,
             phase="train"):
        start_dimension = input_data.shape[-1]
        try:
            iteration_batch = input_data.view(self.config["batch_size"],
                                              self.config["seq_len"],
                                              start_dimension).float()
        except Exception:
            print(f"Error: {input_data.shape}")
            return

        time_embedding = embedding_x
        output, attn_map = self.forward(iteration_batch, time_embedding)

        horizon_output = output[:, -self.config["pred_len"]:, :]
        loss = self.loss(prediction=horizon_output.float(),
                         target=input_target.float())
        if determine_metrics:
            metrics = calc_metrics(output=horizon_output, target=input_target, prefix=f"{phase}_")
            metrics.update({phase + "_loss": float(loss)})
            metrics.update({"loss": float(loss)})

            self.log_dict(metrics, batch_size=iteration_batch.shape[0])

        return loss


class TSRMClassification(TSRM):
    def __init__(self, config: dict):

        super().__init__(config)
        self.loss = nn.functional.cross_entropy

        self.config = config

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict, assign)
        raise NotImplemented

        if self.training:
            target_classes = self.config.get("target_classes")

            for param in self.layers_encoding.parameters():
                param.requires_grad = False
            for attn in [enc._self_attention.parameters() for enc in self.layers_encoding]:
                for param in attn:
                    param.requires_grad = True

    def _run(self, input_data, input_target, meta, embedding=None, determine_metrics=True, calc_real=False):
        try:
            iteration_batch = input_data.view(self.config["batch_size"],
                                              self.config["seq_len"],
                                              self.config["feature_dimension"]).float()
        except Exception:
            print(f"Error: {input_data.shape}")
            return

        mask = build_mask_from_data(iteration_batch).to(input_data.device)
        data = torch.masked_fill(iteration_batch, mask.pow(2).bool(), -1)

        output, classification, attn_map = self.forward(data, embedding)

        loss = self.loss(classification, input_target.to(torch.int64))
        if determine_metrics:
            metrics = calc_metrics(classification_output=classification, classification_target=input_target,
                                   classification_finetune=True, num_classes=self.config["target_classes"])
            metrics.update({"loss": float(loss)})
            self.log_dict(metrics, batch_size=iteration_batch.shape[0])

        return loss


model_dict = {
    "pretrain": TSRMPretrain,
    "imputation": TSRMImputation,
    "forecasting": TSRMForecasting,
    "classification": TSRMClassification
}
