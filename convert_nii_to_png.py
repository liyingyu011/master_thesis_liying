
import os
import nibabel as nib
import numpy as np
from PIL import Image

def normalize_slice(slice_2d):
    # Normalize to 0-255 for visualization
    slice_2d = slice_2d - np.min(slice_2d)
    if np.max(slice_2d) != 0:
        slice_2d = slice_2d / np.max(slice_2d)
    return (slice_2d * 255).astype(np.uint8)

def convert_nii_to_png(task_dir, region, output_dir, axis=2):
    input_path = os.path.join(task_dir, region)
    case_dirs = sorted(os.listdir(input_path))

    a_dir = os.path.join(output_dir, region + "_val", "A")
    b_dir = os.path.join(output_dir, region + "_val", "B")
    os.makedirs(a_dir, exist_ok=True)
    os.makedirs(b_dir, exist_ok=True)

    img_counter = 0

    for case in case_dirs:
        case_path = os.path.join(input_path, case)
        mr_path = os.path.join(case_path, "mr.nii.gz")
        ct_path = os.path.join(case_path, "ct.nii.gz")

        if not os.path.exists(mr_path) or not os.path.exists(ct_path):
            print(f"⚠️ Missing data in: {case_path}")
            continue

        try:
            mr_nii = nib.load(mr_path).get_fdata()
            ct_nii = nib.load(ct_path).get_fdata()
        except Exception as e:
            print(f"❌ Failed to load: {case_path}, reason: {e}")
            continue

        if mr_nii.shape != ct_nii.shape:
            print(f"❗ Shape mismatch in: {case}")
            continue

        for i in range(mr_nii.shape[axis]):
            if axis == 0:
                mr_slice = mr_nii[i, :, :]
                ct_slice = ct_nii[i, :, :]
            elif axis == 1:
                mr_slice = mr_nii[:, i, :]
                ct_slice = ct_nii[:, i, :]
            else:
                mr_slice = mr_nii[:, :, i]
                ct_slice = ct_nii[:, :, i]

            mr_img = Image.fromarray(normalize_slice(mr_slice))
            ct_img = Image.fromarray(normalize_slice(ct_slice))

            mr_img.save(os.path.join(a_dir, f"{region}_{img_counter:05d}.png"))
            ct_img.save(os.path.join(b_dir, f"{region}_{img_counter:05d}.png"))
            img_counter += 1

        print(f"✅ Converted {case}: total slices so far: {img_counter}")

if __name__ == "__main__":
    task_dir = "data/Task1"
    output_dir = "data_prepared"

    convert_nii_to_png(task_dir, "brain", output_dir)
    convert_nii_to_png(task_dir, "pelvis", output_dir)
    print("🎉 All conversions completed.")
