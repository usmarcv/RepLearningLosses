from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch


if __name__ == "__main__":
    out_folders = [Path("save/Linear/path_models/path_vit_small_path_bsz_128_lr_0.0001_size_128/"),
                #    Path("save/linear/cifar2_models/cifar2_lr_5.0_bsz_512_old/"),
                #    Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_new/"),
                #    Path("save/linear/cifar10_models/cifar10_lr_5.0_bsz_512_old/")
                   ]
    fig_folder = Path("figures/confusion_acc")
    fig_folder.mkdir(exist_ok=True)
    # calculate embedding statistics
    for out_folder in out_folders:
        if "path" in out_folder.name:
    
            class_labels = ('bathtub', 'bed', 'chair', 'desk', 'dresser',
                            'monitor', 'night_stand', 'sofa', 'table', 'toilet')

        else:
            # CIFAR-2 labels
            class_labels = ('Cat', 'Dog')

        print(f"[INFO] Processando: {out_folder}")

        preds = torch.argmax(torch.load(out_folder / "preds.pth"), dim=1)
        labels = torch.load(out_folder / "labels.pth")
        
        disp = ConfusionMatrixDisplay.from_predictions(
                    labels, 
                    preds, 
                    display_labels=class_labels, 
                    cmap="Blues"
                )
        
        # plt.title("SINCERE Loss")
        disp.ax_.set_xticklabels(disp.ax_.get_xticklabels(), rotation=45, ha="right")
        disp.figure_.savefig(fig_folder / (out_folder.name + ".pdf"), bbox_inches='tight')
        print(f"[INFO] Plot salvo em {fig_folder / (out_folder.name + '.pdf')}")
