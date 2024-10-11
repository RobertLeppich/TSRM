import torch
import random


def build_mask(batch_size, window_size, feat_dim, /, *, data_batch, mask_size=10, mask_var_length=10, mask_count=4,
               mask_offset=20, pred_len=0, phase="pretrain", task="forecasting", **kwargs):
    mask = torch.zeros((batch_size, window_size, feat_dim), dtype=torch.bool, device=data_batch.device)
    mask[torch.where(data_batch == -1)] = True

    if mask_count > 0:
        area_between = window_size - 2 * mask_offset - mask_count * mask_size

        area_between = area_between // mask_count

        for i in range(mask_count):

            area = mask_offset + i * (area_between + mask_size) + random.randint(mask_size // 2, mask_size)

            mask[:, area:area + mask_size + random.randint(0, mask_var_length), :] = True

    return mask

def build_mask_from_data(data):
    result = []
    for b in data:
        result.append(b.eq(-1))
    return torch.stack(result)

def add_noise(input_batch, noise_count_fraction):
    real_ts = torch.ones(input_batch.shape[0], dtype=torch.bool)
    size = input_batch.shape[1]
    for b in range(input_batch.shape[0]):
        if random.randrange(100) < noise_count_fraction * 100:
            real_ts[b] = 0
            if input_batch.shape[-1] == 1:
                input_batch[b, :, :] = torch.randn_like(input_batch[b, :, :])
            else:
                for i in range(input_batch.shape[-1]):
                    input_batch[b, :, i] = input_batch[b, :, (i+1) % input_batch.shape[-1]]
    return input_batch, real_ts
