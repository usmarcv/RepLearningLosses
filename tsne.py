from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

if __name__ == "__main__":
    out_folders = [#For ModelNet10
                #    Path("inferencetestsib/path_vit_base_path_bsz_128_lr_0.0001_size_224_2025_06_25-13_41_31"),
                   #Path("inferencetestsib/path_dino_vit_base_p_16_path_bsz_128_lr_0.0001_size_224_2025_06_25-17_12_42"),
                   #For ModelNet40
                   Path("save/Linear/path_models/path_dino_vit_base_p_16_path_bsz_128_lr_0.0001_size_224_2025_06_25-23_39_57"),
                   Path("save/Linear/path_models/path_dino_vit_small_p_16_path_bsz_128_lr_0.0001_size_224_2025_06_25-20_45_23")
                   ]
    fig_folder = Path("figures/tsne")
    fig_folder.mkdir(exist_ok=True)

    # ModelNet10 labels
    # class_labels = ('bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet')
    
    
    #ModelNet40 labes
    class_labels = (
        "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
        "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
        "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
        "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
        "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
        "wardrobe", "xbox"
    )



    for out_folder in out_folders:
        print(f"[INFO] Processando: {out_folder}")

        embeds = torch.load(out_folder / "embeds.pth")  # (N, D)
        labels = torch.load(out_folder / "labels.pth")  # (N,)
        labels = labels.numpy()

        proj_embedding = TSNE(n_components=2, perplexity=100, init="random", random_state=31).fit_transform(embeds.numpy())

        fig, ax = plt.subplots(figsize=(12, 10))
        for label in sorted(set(labels)):
            is_label = labels == label
            ax.scatter(proj_embedding[is_label, 0], 
                       proj_embedding[is_label, 1],
                       label=class_labels[int(label)], alpha=0.75, s=40)

        plt.xticks([]) 
        plt.yticks([]) 

        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  #to right legend
        # plt.legend(loc='best', fontsize='small', markerscale=2)

        # ax.set_title(f"t-SNE - {out_folder.name}")
        ax.set_title("SimCLR with DINO B/16")
        plt.tight_layout()
        plt.savefig(fig_folder / (out_folder.name + ".pdf"), bbox_inches='tight')
        print(f"[INFO] Plot salvo em {fig_folder / (out_folder.name + '.pdf')}")


