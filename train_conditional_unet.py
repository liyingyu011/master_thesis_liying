import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])


class PairedImageDataset(Dataset):
    def __init__(self, root_dir, image_size=(256, 256), metadata_field="weight", match_by_slice=False):
        self.root_dir = root_dir
        self.image_size = image_size
        self.metadata_field = metadata_field
        self.match_by_slice = match_by_slice
        self.transform = get_transform(image_size)

        self.sample_list = []  # list of (image_a_path, image_b_path, metadata_value)
        self._gather_samples()

    def _gather_samples(self):
        patient_dirs = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir)
                        if os.path.isdir(os.path.join(self.root_dir, d))]

        for patient_dir in patient_dirs:
            patient_id = os.path.basename(patient_dir)
            metadata_path = os.path.join(patient_dir, f"{patient_id}_info.cfg")
            meta_dict = self._load_metadata(metadata_path)
            metadata_value = float(meta_dict.get(self.metadata_field, 0.0))

            frame01_files = sorted([f for f in os.listdir(patient_dir) if "frame01" in f and f.endswith(".png")])
            frame12_files = sorted([f for f in os.listdir(patient_dir) if "frame12" in f and f.endswith(".png")])

            slices01 = {f.split("slice")[-1]: f for f in frame01_files}
            slices12 = {f.split("slice")[-1]: f for f in frame12_files}

            common_slices = list(set(slices01.keys()) & set(slices12.keys()))

            for slice_id in common_slices:
                image_a_path = os.path.join(patient_dir, slices01[slice_id])
                image_b_path = os.path.join(patient_dir, slices12[slice_id])
                self.sample_list.append((image_a_path, image_b_path, metadata_value))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        try:
            image_a_path, image_b_path, metadata_value = self.sample_list[idx]
            image_a = Image.open(image_a_path).convert("L")
            image_b = Image.open(image_b_path).convert("L")

            image_a = self.transform(image_a)
            image_b = self.transform(image_b)

            # Save for comparison
            gt_b_np = image_b.squeeze().numpy()
            pred_b_np = image_a.squeeze().numpy()  # Placeholder for actual model output
            diff = np.abs(pred_b_np - gt_b_np)

            # Visualize 4 columns: Input A, Generated B, Ground Truth B, Difference
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            for row in range(3):
                axes[row, 0].imshow(image_a.squeeze(), cmap="gray")
                axes[row, 0].set_title("Input (A)")
                axes[row, 1].imshow(pred_b_np, cmap="gray")
                axes[row, 1].set_title("Generated (B)")
                axes[row, 2].imshow(gt_b_np, cmap="gray")
                axes[row, 2].set_title("Ground Truth (B)")
                axes[row, 3].imshow(diff, cmap="hot")
                axes[row, 3].set_title("Difference Heatmap")
                for ax in axes[row]:
                    ax.axis("off")

            plt.tight_layout()
            plt.savefig("comparison_grid_with_diff.png", dpi=300)
            plt.close()

            return {
                "image_a": image_a.float(),
                "image_b": image_b.float(),
                "metadata": torch.tensor(metadata_value, dtype=torch.float)
            }
        except Exception as e:
            print(f"Failed to load sample {idx}: {e}")
            return None

    def _load_metadata(self, path):
        meta = {}
        if not os.path.exists(path):
            return meta
        with open(path, "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":")
                    val = val.strip()
                    try:
                        meta[key.strip()] = float(val) if "." in val else int(val)
                    except:
                        continue
        return meta
