# from pathlib import Path
# import matplotlib.pyplot as plt
# import torch
# import umap

# if __name__ == "__main__":
#     out_folders = [Path("save/Linear/path_models/path_vit_small_path_bsz_128_lr_0.0001_size_128/")]
#     fig_folder = Path("figures/umap")
#     fig_folder.mkdir(exist_ok=True)

#     # ModelNet10 labels
#     class_labels = ('bathtub', 'bed', 'chair', 'desk', 'dresser',
#                     'monitor', 'night_stand', 'sofa', 'table', 'toilet')

#     colors = ['red', 'green', 'blue', 'cyan', 'magenta', 
#           'yellowgreen', 'black', 'plum', 'orange', 'purple']

#     for out_folder in out_folders:
#         print(f"[INFO] Processando: {out_folder}")

#         embeds = torch.load(out_folder / "embeds.pth")  # (N, D)
#         labels = torch.load(out_folder / "labels.pth")  # (N,)
#         labels = labels.numpy()

#         # UMAP projection
#         reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)
#         proj_embedding = reducer.fit_transform(embeds.numpy())

#         fig, ax = plt.subplots(figsize=(8, 6))
#         for label in sorted(set(labels)):
#             is_label = labels == label
#             ax.scatter(proj_embedding[is_label, 0],
#                        proj_embedding[is_label, 1],
#                        label=class_labels[int(label)], alpha=0.6, s=10)

#         ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  # legenda à direita
#         ax.set_title(f"UMAP - {out_folder.name}")
#         plt.tight_layout()
#         plt.savefig(fig_folder / (out_folder.name + ".pdf"), bbox_inches='tight')
#         print(f"[INFO] Plot salvo em {fig_folder / (out_folder.name + '.pdf')}")

from pathlib import Path
import matplotlib.pyplot as plt
import torch
import umap

if __name__ == "__main__":
    out_folders = [Path("save/Linear/path_models/path_vit_small_path_bsz_128_lr_0.0001_size_128/")]
    fig_folder = Path("figures/umap")
    fig_folder.mkdir(exist_ok=True)

    # ModelNet10 labels
    class_labels = ('bathtub', 'bed', 'chair', 'desk', 'dresser',
                    'monitor', 'night_stand', 'sofa', 'table', 'toilet')

    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 
              'yellowgreen', 'black', 'plum', 'orange', 'purple']

    for out_folder in out_folders:
        print(f"[INFO] Processando: {out_folder}")

        embeds = torch.load(out_folder / "embeds.pth")  # (N, D)
        labels = torch.load(out_folder / "labels.pth")  # (N,)
        labels = labels.numpy()

        # UMAP projection
        reducer = umap.UMAP(n_neighbors=100, min_dist=0.1, metric='cosine', random_state=42)
        proj_embedding = reducer.fit_transform(embeds.numpy())

        fig, ax = plt.subplots(figsize=(8, 6))
        handles = []

        for i, label in enumerate(sorted(set(labels))):
            is_label = labels == label
            sc = ax.scatter(
                proj_embedding[is_label, 0],
                proj_embedding[is_label, 1],
                label=class_labels[int(label)],
                color=colors[int(label)],
                s=20,
            )
            handles.append(sc)

        # Legenda no topo central, com estilo parecido ao da imagem
        ax.legend(
            handles=handles,
            labels=[class_labels[int(i)] for i in sorted(set(labels))],
            loc='upper center',
            bbox_to_anchor=(0.5, 1.15),
            ncol=5,
            fancybox=True,
            shadow=True
        )

        ax.set_title(f"UMAP - {out_folder.name}")
        plt.tight_layout()
        plt.savefig(fig_folder / f"{out_folder.name}.pdf", bbox_inches='tight')
        print(f"[INFO] Plot salvo em {fig_folder / (out_folder.name + '.pdf')}")
