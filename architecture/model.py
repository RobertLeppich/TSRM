from typing import Any, Literal, Mapping
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import lightning as pl
from architecture.utils import add_noise, build_mask, build_mask_from_data
from architecture.loss_functions import PreTrainingLoss, ImputationLoss, ForecastingLoss, BinaryClassLoss
from architecture.multiHeadAttention import MultiHeadedAttention
from architecture.metrics import calc_metrics
from embedding.data_embedding import DataEmbedding


class EncodingLayer(pl.LightningModule):

    def __init__(self,
                 h: int,
                 encoding_size: int,
                 dropout: float,
                 data_window_size: int,
                 conv_dims: tuple,  # ((kernel_size, dilation, groups), (...))
                 mask_offset: int,
                 attention_func: Literal["classic", "propsparse", "propsparse_entmax15"],
                 add_pooled_representations: bool = False,
                 clip_data=0,
                 feature_dimension=1,
                 **kwargs):
        super().__init__()
        data_window_size = data_window_size + (2 * clip_data)
        self._self_attention = MultiHeadedAttention(encoding_size=encoding_size, h=h, dropout=dropout,
                                                    attention_func=attention_func,
                                                    feature_dimension=feature_dimension,
                                                    **kwargs)
        self.single_feature_encoding = encoding_size

        encoding_size *= feature_dimension

        self._dropout = nn.Dropout(p=dropout)
        self.representation_layers = nn.ModuleList()
        self.merge_layers = nn.ModuleList()
        self.pools = nn.ModuleList()
        conv_size = 0

        # build representation and merge layers
        for ks, dl, groups in conv_dims:
            if groups == -1:  # depthwise convolution
                groups = encoding_size

            # calculate actual kernel size. ks includes coverage of time series in percent
            kernel_s = int((int(data_window_size * ks) - 1 + dl) / dl)

            stride = max(kernel_s // 4, 1)
            conv_size_layer = self.conv_output_dimension(data_window_size, dl, 0, kernel_s, stride)
            trans_size_layer = self.transpose_output_dimension(conv_size_layer, dl, 0, kernel_s, stride)
            conv_size += conv_size_layer

            # representation CNN
            repr_conv = nn.Conv1d(in_channels=encoding_size,
                                  out_channels=encoding_size,
                                  kernel_size=kernel_s,
                                  stride=stride,
                                  dilation=int(dl),
                                  groups=groups)
            self.representation_layers.append(repr_conv)

            # relative transpose conv to revert convolution
            t_conv = nn.ConvTranspose1d(in_channels=encoding_size,
                                        out_channels=encoding_size,
                                        kernel_size=kernel_s,
                                        output_padding=data_window_size - trans_size_layer,
                                        stride=stride,
                                        dilation=int(dl),
                                        groups=groups)
            self.merge_layers.append(t_conv)

            # max pool layers to extract features from output of cnn layers
            if add_pooled_representations:
                stride = 3
                kernel_s = 3

                self.pools.append(nn.MaxPool1d(kernel_size=kernel_s,
                                               stride=stride))
                conv_size += int((int(conv_size_layer - (kernel_s - 1) - 1) / stride) + 1)

        # resulting matrix length (R)
        self.conv_size = conv_size

        self.fin_ff = nn.Linear(encoding_size * len(conv_dims), encoding_size)

        self.mask_offset = mask_offset
        self._feedForward = PositionalFeedforward(encoding_size)

        self._layerNorm1 = nn.GroupNorm(num_channels=encoding_size, num_groups=feature_dimension)
        self._layerNorm2 = nn.GroupNorm(num_channels=encoding_size, num_groups=feature_dimension)

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
        x = nn.functional.gelu(x).transpose(-1, -2)
        x = self._feedForward(x).transpose(-1, -2)
        x = self._dropout(x)
        x += residual

        global_residual += x

        x = torch.split(x, representations, dim=1)
        x = [self.merge_layers[i](x[i].transpose(-1, -2)).transpose(-1, -2)
             for i in range(len(self.representation_layers))]

        x = torch.cat(x, dim=-1)

        reverse_attn = self.reverse_attn_dim(attn.sum(2), representations, size=data.shape[1])

        x = self.fin_ff(x)
        return x, attn, reverse_attn, global_residual

    def reverse_attn_dim(self, x, representations, size):

        x = (x - x.min()) / (x.max() - x.min())
        splits = torch.split(x, representations, dim=1)

        repr_layers = splits[:len(self.representation_layers)]

        merged = [nn.functional.conv_transpose1d(repr_layers[i].transpose(-1, -2),
                                                 torch.ones((x.shape[-1], x.shape[-1],
                                                             self.representation_layers[i].kernel_size[0]),
                                                            device=x.device) /
                                                 self.representation_layers[i].kernel_size[0],
                                                 dilation=self.representation_layers[i].dilation,
                                                 stride=self.representation_layers[i].stride,
                                                 groups=1).transpose(-1, -2)
                  for i in range(len(repr_layers))]
        merged = [entry if entry.shape[1] == size else
                  nn.functional.pad(entry.transpose(-1, -2), (size - entry.shape[1], 0)).transpose(-1, -2)
                  for entry in merged]
        # c1 -> A,B,C  => A -> c1,c2,c3
        merged = [torch.cat(t, dim=-1).sum(-1).unsqueeze(-1) for t in
                  zip(*[torch.split(entry, 1, dim=-1) for entry in merged])]

        y = torch.cat(merged, dim=-1)
        return (y - y.min()) / (y.max() - y.min())

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
        self.lin1 = nn.Linear(encoding_size, encoding_size * 2)
        self.lin2 = nn.Linear(encoding_size * 2, encoding_size * 2)
        self.lin3 = nn.Linear(encoding_size * 2, encoding_size)

    def forward(self, data):
        data = self.lin1(data.transpose(-1, -2))
        data = self.lin2(data)
        return nn.functional.relu(self.lin3(data)).transpose(-1, -2)


class AttnMapClassification(pl.LightningModule):

    def calc_output_dimension(self, h, ks, stride):
        return int((h - (ks - 1) - 1) / stride) + 1

    def __init__(self, map_size: int, N: int,
                 feature_dimension: int, **kwargs):
        super().__init__()
        self.conv1 = nn.ModuleList(
            [nn.Conv1d(in_channels=N, out_channels=N, kernel_size=3, groups=1) for _ in range(feature_dimension)])
        self.conv2 = nn.ModuleList(
            [nn.Conv1d(in_channels=N, out_channels=N, kernel_size=3, groups=1) for _ in range(feature_dimension)])

        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=3)

        for i in range(4):
            map_size = self.calc_output_dimension(map_size, 3, stride=1 if i % 2 == 0 else 3)
        self.map_size = map_size
        self.channel_lin_1 = nn.ModuleList(
            [nn.Linear(in_features=map_size, out_features=map_size * 2) for _ in range(feature_dimension)])
        self.channel_lin_2 = nn.ModuleList(
            [nn.Linear(in_features=map_size * 2, out_features=1) for _ in range(feature_dimension)])

        self.final_lin1 = nn.Linear(N * feature_dimension, N * feature_dimension)
        self.final_lin2 = nn.Linear(N * feature_dimension, N)
        self.final_lin3 = nn.Linear(N, 1)

    def forward(self, classification_maps):
        # NxFeatures
        classification_maps = [torch.cat(channel, dim=-1) for channel in
                               zip(*[torch.split(entry, 1, dim=-1) for entry in classification_maps])]
        for i in range(len(self.conv1)):
            classification_maps[i] = self.pool1(
                nn.functional.elu(self.conv1[i](classification_maps[i].transpose(1, -1))))
        for i in range(len(self.conv1)):
            classification_maps[i] = self.pool2(nn.functional.elu(self.conv2[i](classification_maps[i])))
        for i in range(len(self.conv1)):
            classification_maps[i] = self.channel_lin_1[i](classification_maps[i])
        for i in range(len(self.conv1)):
            classification_maps[i] = nn.functional.sigmoid(self.channel_lin_2[i](classification_maps[i]))

        result = torch.flatten(torch.cat(classification_maps, dim=-1), 1, 2)
        result = nn.functional.gelu(self.final_lin1(result))
        result = nn.functional.gelu(self.final_lin2(result))
        return self.final_lin3(result).squeeze()


class TimeSeriesRepresentationModel(pl.LightningModule):

    def __init__(self, config: dict, scaler, learning_rate: float = 0.001):
        super().__init__()

        self.scaler = scaler

        self.feature_dimension = config["feature_dimension"]
        self.encoding_size = config["encoding_size"]
        self.data_window_size = config["data_window_size"] + (config.get("clip_data", 0) * 2)
        self.config = config
        self.h = config["h"]
        self.learning_rate = learning_rate
        self.lr = learning_rate
        self.loss = PreTrainingLoss(config)
        self.res_dropout = nn.Dropout(config["dropout"])

        self.encoding_ff = DataEmbedding(config)

        self.layers_encoding = nn.ModuleList([
            EncodingLayer(**config)
            for _ in range(config["N"])
        ])

        self.fin_ff = nn.ModuleList([nn.Linear(self.encoding_size, 1, bias=False)
                                     for _ in range(config["feature_dimension"])])

        final_dim = (self.encoding_size // self.h) * config["N"]
        self.final_dim = final_dim

        self.attn_map_classification = config.get("attn_map_classification", False)
        self.final_classifier = AttnMapClassification(self.layers_encoding[0].conv_size, **config)

    def forward(self, x: torch.Tensor, embedding=None, mask=None):

        encoding = self.encoding_ff(x, embedding)

        encoding = self.clip_data(encoding)

        classifications = []
        reverse_attn_list = []
        residual = None
        # Encoding stack
        for i in range(len(self.layers_encoding)):
            encoding, attn, reverse_attn, residual = self.layers_encoding[i](encoding, residual)

            reverse_attn_list.append([self.de_clip_data(entry)
                                      for entry in torch.split(reverse_attn, self.feature_dimension, dim=-1)][0])

            classifications.append(attn.sum(-2))

        classification_result = self.final_classifier(classifications)

        splits = torch.split(encoding, self.encoding_size, dim=-1)
        splits = [self.fin_ff[i](splits[i]) for i in range(len(splits))]
        encoding = torch.cat(splits, dim=-1)

        attn_map = [torch.cat(entry, -1) for entry in zip(*[torch.split(reverse_attn_list[i], 1, dim=-1)
                                                            for i in range(len(reverse_attn_list))])]

        return self.de_clip_data(encoding), nn.functional.sigmoid(classification_result), attn_map

    def clip_data(self, data):
        if (clip_range := self.config.get("clip_data", 0)) > 0:
            start = torch.ones_like(data[:, :clip_range, :]) * -1  # torch.flip(data[:, :clip_range, :], dims=[1])
            end = torch.ones_like(data[:, -clip_range:, :]) * -1  # torch.flip(data[:, -clip_range:, :], dims=[1])
            return torch.cat([start, data, end], dim=1)
        return data

    def de_clip_data(self, data):
        if (clip_range := self.config.get("clip_data", 0)) > 0:
            return data[:, clip_range:-clip_range, :]
        return data

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr or self.learning_rate)

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
            input_data, input_target, meta, embedding = input_batch

        loss = self._run(input_data, input_target, meta, embedding, determine_metrics=False)
        return loss

    def validation_step(self, input_batch, idx):

        meta = None
        embedding = None

        if len(input_batch) == 2:
            input_data, input_target = input_batch
        elif len(input_batch) == 3:
            input_data, input_target, meta = input_batch
        else:
            input_data, input_target, meta, embedding = input_batch

        loss = self._run(input_data, input_target, meta, embedding, determine_metrics=True)
        return loss

    def test_step(self, input_batch, idx):
        meta = dict()
        embedding = None

        if len(input_batch) == 2:
            input_data, input_target = input_batch
        elif len(input_batch) == 3:
            input_data, input_target, meta = input_batch
        else:
            input_data, input_target, meta, embedding = input_batch

        loss = self._run(input_data, input_target, meta, embedding, determine_metrics=True,
                         calc_real=self.config.get("minmax_scaled", False))
        return loss

    def _run(self, input_data, input_target, meta, embedding=None, determine_metrics=True, calc_real=False):
        # reshape input
        try:
            iteration_batch = input_data.view(self.config["batch_size"],
                                              self.config["data_window_size"],
                                              self.config["feature_dimension"]).float()
        except Exception:
            print(f"Error: {input_data.shape}")
            return

        # build random mask
        mask = meta if isinstance(meta, torch.Tensor) else build_mask(*iteration_batch.shape, data=iteration_batch,
                                                                      **self.config).to(self.device)

        # add noise
        data, real_ts = add_noise(iteration_batch.clone(),
                                  noise_count_fraction=self.config["noise_count_fraction"])

        # mask data
        data = torch.masked_fill(data, mask.pow(2).bool(), -1)

        output, classification, attn_map = self.forward(data, embedding, mask)

        loss = self.loss(prediction_imputation=output,
                         target_imputation=iteration_batch,
                         prediction_binary=classification,
                         target_binary=real_ts.to(self.device), mask=mask.eq(1))
        if determine_metrics:
            metrics = calc_metrics(output=output[real_ts], target=iteration_batch[real_ts],
                                   classification_output=classification,
                                   classification_target=real_ts.to(self.device).long(),
                                   mask=mask.eq(1)[real_ts], scaler=self.scaler, meta=meta, calc_real=calc_real)
            metrics.update({"loss": float(loss)})
            self.log_dict(metrics, batch_size=iteration_batch.shape[0])
            self.logger.log_metrics(metrics)
            return metrics
        return loss


class EncoderConvFinetuneImputation(TimeSeriesRepresentationModel):
    def __init__(self, config: dict, scaler):

        super().__init__(config, scaler)
        self.loss = ImputationLoss(config)

    def _run(self, input_data, input_target, meta, embedding=None, determine_metrics=True, calc_real=False):
        start_dimension = input_data.shape[-1]
        try:
            iteration_batch = input_data.view(self.config["batch_size"],
                                              self.config["data_window_size"],
                                              start_dimension).float()
        except Exception:
            print(f"Error: {input_data.shape}")
            return

        mask = meta if isinstance(meta, torch.Tensor) else build_mask_from_data(iteration_batch.detach().cpu()).to(
            input_data.device)
        data = torch.masked_fill(iteration_batch, mask.pow(2).bool(), -1)

        output, classification, attn_map = self.forward(data, embedding)

        loss = self.loss(prediction=output,
                         target=input_target,
                         mask=mask.eq(1))
        if determine_metrics:
            metrics = calc_metrics(output=output, target=input_target,
                                   mask=mask.eq(1), scaler=self.scaler, meta=meta, calc_real=calc_real)
            metrics.update({"loss": float(loss)})
            self.log_dict(metrics, batch_size=iteration_batch.shape[0])

        return loss


class EncoderConvFinetuneForecasting(TimeSeriesRepresentationModel):
    def __init__(self, config: dict, scaler):

        super().__init__(config, scaler)
        self.loss = ForecastingLoss(config)

        for param in self.layers_classifier.parameters():
            param.requires_grad = False
        for param in self.final_classifier.parameters():
            param.requires_grad = False

    def _run(self, input_data, input_target, meta, embedding=None, determine_metrics=True, calc_real=False):
        start_dimension = input_data.shape[-1]
        try:
            iteration_batch = input_data.view(self.config["batch_size"],
                                              self.config["data_window_size"],
                                              start_dimension).float()
        except Exception:
            print(f"Error: {input_data.shape}")
            return

        mask = build_mask_from_data(iteration_batch).to(input_data.device)
        data = torch.masked_fill(iteration_batch, mask.pow(2).bool(), -1)

        output, classification, attn_map = self.forward(data, embedding)
        horizon_output = output[:, -self.config["horizon"]:, -1].unsqueeze(-1)
        loss = self.loss(prediction=horizon_output,
                         target=input_target)
        if determine_metrics:

            metrics = calc_metrics(output=horizon_output, target=input_target,
                                   mask=mask[:, -self.config["horizon"]:, -1].unsqueeze(-1),
                                   scaler=self.scaler, meta=meta, calc_real=calc_real)
            metrics.update({"loss": float(loss)})

            if (second_horizon := self.config.get("horizon2", None)) is not None:
                horizon_output2 = output[:, -self.config["horizon"]:-(self.config["horizon"] - second_horizon),
                                  -1].unsqueeze(-1)

                metrics2 = calc_metrics(output=horizon_output2, target=input_target[:, :-second_horizon, :],
                                        mask=mask[:, -self.config["horizon"]:-(self.config["horizon"] - second_horizon),
                                             -1].unsqueeze(-1),
                                        scaler=self.scaler, meta=meta, calc_real=calc_real, prefix=f"{second_horizon}_")
                metrics.update(metrics2)
            self.log_dict(metrics, batch_size=iteration_batch.shape[0])

        return loss


class EncoderConvFinetuneClassification(TimeSeriesRepresentationModel):
    def __init__(self, config: dict, scaler):

        super().__init__(config, scaler)
        self.loss = nn.functional.cross_entropy

        self.config = config

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict, assign)
        target_classes = self.config.get("target_classes")
        N = self.config.get("N")
        # replace final classifier
        if not self.attn_map_classification:
            self.final_classifier = FinalClassifier(self.final_dim, self.config.get("target_classes"), **self.config)
        else:

            self.final_classifier.channel_lin_2 = nn.ModuleList(
                [nn.Linear(in_features=self.final_classifier.map_size * 2, out_features=target_classes)
                 for _ in range(self.config["feature_dimension"])])

            self.final_classifier.final_lin1 = nn.Linear(N * target_classes * self.config["feature_dimension"],
                                                         N * target_classes * 2)
            self.final_classifier.final_lin2 = nn.Linear(N * target_classes * 2, N * target_classes)
            self.final_classifier.final_lin3 = nn.Linear(N * target_classes, target_classes)

        for param in self.layers_encoding.parameters():
            param.requires_grad = False

    def _run(self, input_data, input_target, meta, embedding=None, determine_metrics=True, calc_real=False):
        start_dimension = input_data.shape[-1]
        try:
            iteration_batch = input_data.view(self.config["batch_size"],
                                              self.config["data_window_size"],
                                              start_dimension).float()
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
    "pretrain": TimeSeriesRepresentationModel,
    "finetune_imputation": EncoderConvFinetuneImputation,
    "finetune_forecasting": EncoderConvFinetuneForecasting,
    "finetune_classification": EncoderConvFinetuneClassification
}
