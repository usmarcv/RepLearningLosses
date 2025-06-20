from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

if __name__ == "__main__":
    out_folders = [Path("save/Linear/path_models/path_vit_small_path_bsz_128_lr_0.0001_size_128/")]
    fig_folder = Path("figures/tsne")
    fig_folder.mkdir(exist_ok=True)

    # ModelNet10 labels
    class_labels = ('bathtub', 'bed', 'chair', 'desk', 'dresser',
                    'monitor', 'night_stand', 'sofa', 'table', 'toilet')

    for out_folder in out_folders:
        print(f"[INFO] Processando: {out_folder}")

        embeds = torch.load(out_folder / "embeds.pth")  # (N, D)
        labels = torch.load(out_folder / "labels.pth")  # (N,)
        labels = labels.numpy()

        proj_embedding = TSNE(perplexity=50, init="random").fit_transform(embeds.numpy())

        fig, ax = plt.subplots(figsize=(8, 6))
        for label in sorted(set(labels)):
            is_label = labels == label
            ax.scatter(proj_embedding[is_label, 0], 
                       proj_embedding[is_label, 1],
                       label=class_labels[int(label)], alpha=0.6, s=10)

        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  # legenda à direita
        # plt.legend(loc='best', fontsize='small', markerscale=2)

        ax.set_title(f"t-SNE - {out_folder.name}")
        plt.tight_layout()
        plt.savefig(fig_folder / (out_folder.name + ".pdf"), bbox_inches='tight')
        print(f"[INFO] Plot salvo em {fig_folder / (out_folder.name + '.pdf')}")

