import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme("paper")


def plot_attention_weights(line1, line2, line3=None, mask=None, feature=0, attn_map=None, title=None, n_aggregation="sum",
                           threshold=.85, y_label=None, batches_to_plot=5):
    line1 = line1[:, :, feature].detach().cpu()
    line2 = line2[:, :, feature].detach().cpu()

    if mask is not None:
        mask = mask.bool().detach().cpu().numpy()[:, :, feature]

    if line3 is not None:
        line3 = line3[:, :, feature].detach().cpu()

    for batch in range(min(len(line1), batches_to_plot)):
        if attn_map is not None:
            batch_attn_map = attn_map[feature][batch].detach().cpu().numpy()
            if n_aggregation == "sum":
                batch_attn_map = batch_attn_map.sum(-1)
            else:
                batch_attn_map = batch_attn_map[:, n_aggregation]
            batch_attn_map = (batch_attn_map - np.min(batch_attn_map)) / (np.max(batch_attn_map) - np.min(batch_attn_map))
        else:
            batch_attn_map = None

        plt.figure()
        fig, ax = plt.subplots(1,1, figsize=(18, 5))

        if mask is not None:
            masked = line1[batch, :].clone()
            masked[~mask[batch]] = np.nan
            line1[batch, mask[batch]] = np.nan

        ax.plot(line1[batch], color="green", label="Input")
        if mask is not None:
            ax.plot(masked, color="green", label="Masked target", linestyle="dotted", linewidth=2)

        ax.plot(line2[batch], color="red", label="Output")
        if line3 is not None:
            ax.plot(line3[batch], color="blue", label="Custom")

        if attn_map is not None:
            for i in range(line1.shape[1]):
                ax.axvspan(i-0.5, i+0.5, color="red", alpha=max(0, batch_attn_map[i] - threshold))
        if title is not None:
            ax.set_title(title, fontsize=18)

        ax.set_ylabel(y_label or "value", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=18)

        plt.tight_layout()
        plt.show()
        plt.close()