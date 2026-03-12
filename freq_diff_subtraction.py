import os
import re
import argparse
import time
import numpy as np
import tifffile as tiff
import ants
import torch
import torch.nn.functional as F
from skimage.morphology import disk

# Setup device
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA device.")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS device.")
else:
    device = torch.device('cpu')
    print("Using CPU device.")

def apply_saved_transform(fixed_reference, new_moving_image, transform_path):
    if not os.path.exists(transform_path):
        raise FileNotFoundError(f"Transform file not found: {transform_path}")

    spacing = (0.92, 0.92, 0.92)
    if isinstance(new_moving_image, np.ndarray):
        new_moving_image = ants.from_numpy(new_moving_image, spacing=spacing)
    if isinstance(fixed_reference, np.ndarray):
        fixed_reference = ants.from_numpy(fixed_reference, spacing=spacing)
        
    transformed_image = ants.apply_transforms(
        fixed=fixed_reference,
        moving=new_moving_image,
        transformlist=[transform_path],
        interpolator='linear'
    )
    return transformed_image

def get_matched_files(folder1, folder2):
    def get_id_map(folder, pattern):
        if not os.path.exists(folder):
             return {}
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff'))]
        id_map = {}
        for f in files:
            m = re.search(pattern, f)
            if m:
                id_map[int(m.group(1))] = os.path.join(folder, f)
        return id_map

    pattern_generic = r"(\d+)(?=\.\w+$)" 
    map1 = get_id_map(folder1, pattern_generic)
    map2 = get_id_map(folder2, pattern_generic)
    common_ids = sorted(list(set(map1.keys()) & set(map2.keys())))
    
    if not common_ids:
        print("Warning: No matching slice IDs found between the two folders!")
        return [], [], []

    print(f"MATCHING: Found {len(common_ids)} common slices.")
    files1 = [map1[i] for i in common_ids]
    files2 = [map2[i] for i in common_ids]
    return files1, files2, common_ids

def load_tiff_sequence(folder, str_pt=999, slices=50, files_list=None):
    if files_list is not None:
        files = files_list
    else:
        files = sorted(os.listdir(folder))
        files = [f for f in files if f.endswith('.tif') or f.endswith('.tiff')]
        if len(files) == 0:
            raise FileNotFoundError(f"No TIFF files found in {folder}")
        parsed_files = []
        for f in files:
            parts = f.replace('.tif', '').replace('.tiff', '').split('_')
            try:
                idx = int(parts[-1])
                parsed_files.append((idx, f))
            except ValueError:
                pass
        parsed_files.sort(key=lambda x: x[0])
        files = [os.path.join(folder, f[1]) for f in parsed_files if str_pt <= f[0] < str_pt + slices]

    if len(files) == 0:
         print(f"Warning: No valid TIFF files found to load.")
         return None

    volume = np.stack([tiff.imread(f) for f in files], axis=0)
    return ants.from_numpy(volume.astype(np.float32))

def hist_diff_ints(stack_33p1keV, stack_33p2keV, diff_threshold):
    flat_33p1keV = torch.flatten(torch.tensor(stack_33p1keV, device=device, dtype=torch.float32))
    flat_33p2keV = torch.flatten(torch.tensor(stack_33p2keV, device=device, dtype=torch.float32))

    min_val = torch.min(flat_33p1keV.min(), flat_33p2keV.min()).item()
    max_val = torch.max(flat_33p1keV.max(), flat_33p2keV.max()).item()
    bins = torch.linspace(min_val, max_val, steps=256, device=device)

    hist_33p1keV = torch.histc(flat_33p1keV, bins=256, min=min_val, max=max_val)
    hist_33p2keV = torch.histc(flat_33p2keV, bins=256, min=min_val, max=max_val)

    hist_diff = hist_33p2keV - hist_33p1keV
    significant_indices = torch.where(hist_diff > diff_threshold)[0]

    if len(significant_indices) == 0:
        return bins[0].item(), bins[-1].item()
    return bins[significant_indices.min()].item(), bins[significant_indices.max()].item()

def subtraction(vol_33_1, vol_33_2, lower_roi, upper_roi):
    try:
        moving = torch.tensor(vol_33_2.numpy(), device=device, dtype=torch.float32)
        fixed = torch.tensor(vol_33_1.numpy(), device=device, dtype=torch.float32)
    except:
        moving = torch.tensor(vol_33_2, device=device, dtype=torch.float32)
        fixed = torch.tensor(vol_33_1, device=device, dtype=torch.float32)

    moving = torch.where((moving >= lower_roi) & (moving <= upper_roi), moving, torch.tensor(0.0, device=device))
    fixed = torch.where((fixed >= lower_roi) & (fixed <= upper_roi), fixed, torch.tensor(0.0, device=device))

    difference = moving - fixed
    difference_np = difference.cpu().numpy()
    difference_ants = ants.from_numpy(difference_np)
    return difference_ants

def erosion_process2d_gpu(binary_mask_np, n=10, iterations=1, restore_size=True):
    assert binary_mask_np.ndim == 3, "Input must be a 3D binary mask."
    binary_mask = torch.tensor(binary_mask_np, dtype=torch.float32, device=device)

    selem = disk(n // 2).astype(np.float32)
    kernel_sum = selem.sum()
    selem_tensor = torch.tensor(selem, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    padding = selem.shape[0] // 2
    processed = []

    for i in range(binary_mask.shape[0]):
        slice_ = binary_mask[i:i+1].unsqueeze(0)

        eroded = slice_
        for _ in range(iterations):
            conv_result = F.conv2d(eroded, selem_tensor, padding=padding)
            eroded = (conv_result == kernel_sum).float()

        if restore_size:
            dilated = eroded
            for _ in range(iterations):
                conv_result = F.conv2d(dilated, selem_tensor, padding=padding)
                dilated = (conv_result >= 1).float()
            eroded = dilated

        processed.append(eroded.squeeze().cpu().numpy().astype(np.uint8))

    return np.stack(processed, axis=0)

def subero_in_chunks(vol_33_1, vol_33_2, vol_33_2_reg, str_pt, roi, chunk_size=10, low_thresh=5e-6, output_dir=".", n=10, iterations=1, slice_indices=None):
    num_slices = vol_33_1.shape[0]
    
    for start_slice in range(0, num_slices, chunk_size):
        end_slice = min(start_slice + chunk_size, num_slices)

        chunk1 = vol_33_1[start_slice:end_slice]
        chunk2 = vol_33_2[start_slice:end_slice]
        chunk2_reg = vol_33_2_reg[start_slice:end_slice]

        lower_roi, upper_roi = hist_diff_ints(chunk1, chunk2, diff_threshold=roi)
        print(f'[{(end_slice - 1)/(num_slices - 1)*100 if num_slices > 1 else 100:.1f}% | ROI={roi:.1e}] {lower_roi:.2e}, {upper_roi:.2e}')

        ants_chunk1 = ants.from_numpy(chunk1)
        ants_chunk2 = chunk2_reg

        difference_ants = subtraction(ants_chunk1, ants_chunk2, lower_roi=lower_roi, upper_roi=upper_roi)

        binary_mask = ants.threshold_image(difference_ants, low_thresh=low_thresh, high_thresh=np.inf)
        binary_mask_np = binary_mask.numpy()

        smooth_mask_np2 = erosion_process2d_gpu(binary_mask_np, n=n, iterations=iterations)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        current_chunk_indices = slice_indices[start_slice:end_slice] if slice_indices is not None else None
            
        for i in range(smooth_mask_np2.shape[0]):
            if current_chunk_indices is not None:
                save_index = current_chunk_indices[i]
            else:
                save_index = str_pt + start_slice + i
                
            save_path = os.path.join(output_dir, f"slice_{save_index}.tiff")
            tiff.imwrite(save_path, smooth_mask_np2[i].astype(np.float32), dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="Pipeline for processing v1 (2 images + mat)")
    parser.add_argument("--folder1", type=str, required=True, help="Path to fixed image folder")
    parser.add_argument("--folder2", type=str, required=True, help="Path to moving image folder")
    parser.add_argument("--transform", type=str, required=True, help="Path to .mat transform file")
    
    parser.add_argument("--chunk_size", type=int, default=32, help="Number of slices per chunk")
    parser.add_argument("--roi", type=float, default=1e3, help="ROI threshold for difference")
    parser.add_argument("--low_thresh", type=float, default=5e-6, help="Low threshold for subtraction difference")
    parser.add_argument("--n", type=int, default=10, help="Erosion kernel size")
    parser.add_argument("--iterations", type=int, default=1, help="Number of erosion iterations")
    parser.add_argument("--start-slice", type=int, default=0, help="Start slice index")
    parser.add_argument("--end-slice", type=int, default=2047, help="End slice index")
    
    args = parser.parse_args()
    
    files1, files2, common_ids = get_matched_files(args.folder1, args.folder2)
    
    # Filter by slice interval
    filtered_indices = [i for i, cid in enumerate(common_ids) if args.start_slice <= cid <= args.end_slice]
    common_ids = [common_ids[i] for i in filtered_indices]
    files1 = [files1[i] for i in filtered_indices]
    files2 = [files2[i] for i in filtered_indices]
    
    if not common_ids:
        print("Error: No matching slices found within the specified range.")
        return
        
    print(f"Processing matched {len(common_ids)} slices.")
    # Set up output directory next to folder1
    folder1_path = os.path.normpath(args.folder1)
    parent_dir = os.path.dirname(folder1_path)
    base_name = os.path.basename(folder1_path)
    output_dir = os.path.join(parent_dir, f"{base_name}_subtracted")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"Output directory: {output_dir}")
    
    for start_idx in range(0, len(common_ids), args.chunk_size):
        end_idx = min(start_idx + args.chunk_size, len(common_ids))
        
        batch_files1 = files1[start_idx:end_idx]
        batch_files2 = files2[start_idx:end_idx]
        batch_indices = common_ids[start_idx:end_idx]
        
        print(f"[Batch {start_idx}-{end_idx}] Loading {len(batch_indices)} matched slices...")
        
        vol_33_1 = load_tiff_sequence(args.folder1, files_list=batch_files1)
        vol_33_2 = load_tiff_sequence(args.folder2, files_list=batch_files2)
        
        if vol_33_1 is None or vol_33_2 is None:
            print(f"Skipping empty batch {start_idx}-{end_idx}")
            continue
            
        start_time = time.time()
        vol_33_2_reg = apply_saved_transform(vol_33_1, vol_33_2, args.transform)
        print(f"  Transformation time: {time.time() - start_time:.2f} seconds")
        
        # Shape matching
        v1_shape = vol_33_1.shape
        v2reg_shape = vol_33_2_reg.shape
        
        if v1_shape != v2reg_shape:
            print("Warning: Shape mismatch in batch!")
            np_1 = vol_33_1.numpy()
            np_2 = vol_33_2_reg.numpy()
            
            min_z = min(np_1.shape[0], np_2.shape[0])
            min_y = min(np_1.shape[1], np_2.shape[1])
            min_x = min(np_1.shape[2], np_2.shape[2])
            
            np_1 = np_1[:min_z, :min_y, :min_x]
            np_2 = np_2[:min_z, :min_y, :min_x]
            
            vol_33_1 = ants.from_numpy(np_1)
            vol_33_2_reg = ants.from_numpy(np_2)
        
        vol_33_1_np = vol_33_1.numpy()
        vol_33_2_np = vol_33_2.numpy()[:vol_33_1_np.shape[0], :vol_33_1_np.shape[1], :vol_33_1_np.shape[2]]
        
        subero_in_chunks(
            vol_33_1_np,
            vol_33_2_np,
            vol_33_2_reg,
            str_pt=0, 
            roi=args.roi,
            chunk_size=args.chunk_size,
            low_thresh=args.low_thresh,
            output_dir=output_dir,
            n=args.n,
            iterations=args.iterations,
            slice_indices=batch_indices
        )
        print(f"Completed batch {start_idx}-{end_idx}")

if __name__ == "__main__":
    main()
