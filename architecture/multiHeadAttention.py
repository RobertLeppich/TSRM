import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Literal
from architecture.entmax import entmax15


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        for i in range(scores.shape[1]):
            scores[:, i, :, :] = scores[:, i, :, :].masked_fill(mask.bool(), -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


def attention_entmax15(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    p_attn = entmax15(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=True, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, mask, dropout):
        self.dropout = dropout
        B, H, L_Q, D = queries.shape
        _, _, L_K, _ = keys.shape

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, mask)

        return context.transpose(2, 1).contiguous(), attn

class ProbEntmax15Attention(ProbAttention):
    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = entmax15(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)


class MultiHeadedAttention(nn.Module):

    def __init__(self, h: int, encoding_size: int,
                 attention_func: Literal["classic", "propsparse", "entmax15", "propsparse_entmax15", "fourier1" ]="classic",
                 dropout=0.1, feature_dimension=1,
                 **kwargs):

        super(MultiHeadedAttention, self).__init__()

        self.h = h
        self.feature_dimension = feature_dimension
        self.encoding_size = encoding_size

        self.lin_query = nn.ModuleList([nn.Linear(encoding_size, encoding_size) for _ in range(feature_dimension)])
        self.lin_key = nn.ModuleList([nn.Linear(encoding_size, encoding_size) for _ in range(feature_dimension)])
        self.lin_values = nn.ModuleList([nn.Linear(encoding_size, encoding_size) for _ in range(feature_dimension)])
        self.lin_fin = nn.ModuleList([nn.Linear(encoding_size, encoding_size) for _ in range(feature_dimension)])

        self.dropout = nn.Dropout(p=dropout)
        if attention_func == "propspares":
            self.attention = ProbAttention(output_attention=True)
        elif attention_func == "propsparse_entmax15":
            self.attention = ProbEntmax15Attention(output_attention=True)
        elif attention_func == "entmax15":
            self.attention = attention_entmax15
        else:
            self.attention = attention

    def forward(self, query, key, value, mask=None):

        result, attns = [], []
        queries, keys, values = (torch.split(query, self.encoding_size, -1),
                                     torch.split(key, self.encoding_size, -1),
                                     torch.split(value, self.encoding_size, -1))
        for i in range(self.feature_dimension):
            i_res, i_attn = self.forward_single(queries[i], keys[i], values[i],
                                                    self.lin_query[i], self.lin_key[i], self.lin_values[i],
                                                    self.attention, self.lin_fin[i], mask=mask)
            result.append(i_res)
            if i_attn is not None:
                attns.append(i_attn.unsqueeze(-1))

        return torch.cat(result, -1), torch.cat(attns, -1) if len(attns) > 0 else None

    def forward_single(self, query, key, value, lin_query, lin_key, lin_values, attention, lin_fin, mask=None):
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = lin_query(query).view(nbatches, -1, self.h, self.encoding_size // self.h).transpose(1, 2)
        key = lin_key(key).view(nbatches, -1, self.h, self.encoding_size // self.h).transpose(1, 2)
        value = lin_values(value).view(nbatches, -1, self.h, self.encoding_size // self.h).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.encoding_size)

        return lin_fin(x), attn.sum(dim=1) if attn is not None else None