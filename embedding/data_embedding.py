from torch import nn
import torch
import math


class PositionalEncoding2(nn.Module):

    def __init__(self, d_embedd: int, batch_size: int, data_window_size: int):
        super().__init__()

        position = torch.arange(data_window_size).unsqueeze(1)
        pe = torch.zeros(batch_size, data_window_size, d_embedd)
        for i in range(d_embedd):
            pe[:, :, i] = (0.5 + 0.5 * torch.sin(position * (i/d_embedd))).squeeze(-1)
        self.register_buffer('pe', pe)

    def forward(self):
        return self.pe


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = 0.5 + 0.5 * torch.sin(position * div_term)
        pe[:, 1::2] = 0.5 + 0.5 * torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, freq='t'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        if freq == 't':
            self.minute_embed = nn.Embedding(minute_size, d_model)
        self.hour_embed = nn.Embedding(hour_size, d_model)
        self.weekday_embed = nn.Embedding(weekday_size, d_model)
        self.day_embed = nn.Embedding(day_size, d_model)
        self.month_embed = nn.Embedding(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        t = hour_x + weekday_x + day_x + month_x + minute_x
        t = (t - t.min()) / (t.max() - t.min())
        return t


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='t'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        t = self.embed(x.to(torch.float32))
        t = (t - t.min()) / (t.max() - t.min())
        return t + t.min().abs()

class DataEmbedding(nn.Module):
    def __init__(self, config, freq='t', dropout=0.1):
        c_in, d_model = config["feature_dimension"], config["encoding_size"]
        super(DataEmbedding, self).__init__()

        self.value_embedding = nn.ModuleList([nn.Linear(1, d_model, bias=False)
                                                  for _ in range(config["feature_dimension"])])

        self.pos_encoding_activated = config.get("add_embedding", False)
        if self.pos_encoding_activated:
            # self.position_embedding = PositionalEmbedding(d_model=d_model)
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, freq=freq)
            #self.temporal_embedding = TemporalEmbedding(d_model=d_model, freq=freq)
            #self.positional_embedding = PositionalEmbedding(d_model=d_model, max_len=config["data_window_size"])
       # self.dropout = nn.Dropout(dropout)


    def forward(self, x, x_mark):
        result = []
        for i in range(x.shape[-1]):
            x_i = self.value_embedding[i](x[:, :, [i]])
            if self.pos_encoding_activated:
                x_i[:, :, :] += self.temporal_embedding(x_mark)
                # x_i[:, :, :] += self.positional_embedding(x[:, :, [i]])
            result.append(x_i)
        return torch.cat(result, dim=-1)
