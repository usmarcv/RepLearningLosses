import os
import csv
import argparse

def get_obj_id(filename):
    # Ex: "bathtub_0020_4.jpg" → "bathtub_0020"
    base = os.path.basename(filename)
    parts = base.split("_")
    return "_".join(parts[:2])

def generate_csv(root_dir, split, output_csv):
    rows = []
    class_dirs = sorted(os.listdir(os.path.join(root_dir, split)))

    for cls in class_dirs:
        cls_path = os.path.join(root_dir, split, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in sorted(os.listdir(cls_path)):
            if not fname.endswith(".jpg"):
                continue
            path = os.path.join(split, cls, fname)
            obj_id = get_obj_id(fname)
            label = cls
            rows.append([path, obj_id, label])

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'obj_id', 'label'])
        writer.writerows(rows)

    print(f"[INFO] Saved CSV to: {output_csv}, total samples: {len(rows)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='Ex: ModelNet10/')
    parser.add_argument('--split', type=str, choices=['train', 'test'], required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    args = parser.parse_args()

    generate_csv(args.root_dir, args.split, args.output_csv)

