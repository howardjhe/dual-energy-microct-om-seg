import os
import re
import argparse
import shutil
import gc
import numpy as np
import tifffile as tiff
import ants
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def clear_memory():
    """
    Clears Python objects, triggers Garbage Collection, 
    and attempts to release system-level cached memory.
    """
    try:
        ants.utils.process_utils.terminate()
    except:
        pass

    gc.collect()
    print("Memory allocation cleared.")

def load_tiff_sequence(folder, start_slice, end_slice):
    """
    Load a specified range of a TIF sequence based on slice numbers in filenames.
    """
    file_list = [f for f in os.listdir(folder) if f.endswith(".tiff") or f.endswith(".tif")]
    
    if len(file_list) == 0:
        raise ValueError(f"No TIFF files found in {folder}")

    parsed_files = []
    pattern_generic = r"(\d+)(?=\.\w+$)"
    for f in file_list:
        m = re.search(pattern_generic, f)
        if m:
            parsed_files.append((int(m.group(1)), os.path.join(folder, f)))
    
    parsed_files.sort(key=lambda x: x[0])
    
    if start_slice is not None and end_slice is not None:
        filtered_files = [f_path for idx, f_path in parsed_files if start_slice <= idx <= end_slice]
    else:
        filtered_files = [f_path for idx, f_path in parsed_files]
        
    if not filtered_files:
        raise ValueError(f"No TIFF files found in {folder} within the specified range.")

    def load_image(file):
        return tiff.imread(file)
    
    with ThreadPoolExecutor() as executor:
        volume = list(executor.map(load_image, filtered_files))
    
    # We also return the actual starting index for the transform name
    m = re.search(pattern_generic, filtered_files[0])
    actual_start = int(m.group(1)) if m else 0
    
    return np.stack(volume, axis=0).astype(np.float32), actual_start, len(filtered_files)

def process_registration_pair(folder_p1, folder_p2, start_slice=None, end_slice=None, output_dir=None):
    """
    Process two specifically provided reconstructed image sequence folders.
    folder_p1 serves as the fixed image (e.g. 33.1 keV),
    folder_p2 serves as the moving image (e.g. 33.2 keV).
    
    Returns (processed_image_folder_address, mat_address)
    """
    folder_p1 = Path(folder_p1)
    folder_p2 = Path(folder_p2)

    if output_dir is None:
        output_dir = folder_p1.parent / f"{folder_p1.name}_registered"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    clear_memory()

    print(f"\n--- Processing Pair ---")
    print(f"Fixed Folder (P1): {folder_p1}")
    print(f"Moving Folder (P2): {folder_p2}")
    
    vol_p1, actual_start, num_slices_1 = load_tiff_sequence(str(folder_p1), start_slice, end_slice)
    vol_p2, _, num_slices_2 = load_tiff_sequence(str(folder_p2), start_slice, end_slice)
    
    # Use the minimum number of slices loaded to ensure shapes match
    num_slices = min(num_slices_1, num_slices_2)
    vol_p1 = vol_p1[:num_slices]
    vol_p2 = vol_p2[:num_slices]
    
    print(f"Start Slice: {actual_start}, Num Slices: {num_slices}")

    fixed_image = ants.from_numpy(vol_p1)
    moving_image = ants.from_numpy(vol_p2)
    print('Fixed and moving 3D images loaded!')

    print('Performing 3D registration...')
    registration = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform='Rigid',
        metric='Mattes',
    )

    # 1. Save Warped Image (.nii)
    warped_moving_image = registration['warpedmovout']
    save_filename = f"str{actual_start}_slices{num_slices}.nii"
    processed_image_path = output_dir / save_filename
    ants.image_write(warped_moving_image, str(processed_image_path))
    
    # 2. Save Transformation Matrix (.mat)
    mat_dest = None
    if 'fwdtransforms' in registration:
        mat_src = registration['fwdtransforms'][0]
        mat_dest = output_dir / f"str{actual_start}_slices{num_slices}_transform.mat"
        shutil.copy(mat_src, str(mat_dest))
        print(f'Transformation matrix saved to: {mat_dest}')

    print(f'3D registration image saved to: {processed_image_path}')

    processed_image_folder_address = str(output_dir)
    mat_address = str(mat_dest) if mat_dest else None
    
    return processed_image_folder_address, mat_address

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ANTs 3D rigid registration processing for two specified folders.")
    parser.add_argument("folder_p1", type=str, help="Fixed image folder path (e.g., 33.1 keV rec folder)")
    parser.add_argument("folder_p2", type=str, help="Moving image folder path (e.g., 33.2 keV rec folder)")
    parser.add_argument("--start-slice", type=int, default=None, help="Start slice index.")
    parser.add_argument("--end-slice", type=int, default=None, help="End slice index.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output save path. Defaults to a new directory beside folder_p1.")
    
    args = parser.parse_args()
    
    try:
        output_folder, mat_path = process_registration_pair(
            folder_p1=args.folder_p1,
            folder_p2=args.folder_p2,
            start_slice=args.start_slice,
            end_slice=args.end_slice,
            output_dir=args.output_dir
        )
        print("\n=== Processing Results ===")
        print(f"Processed Image Folder: {output_folder}")
        print(f"Transformation Matrix: {mat_path}")
        
    except Exception as e:
        print(f"Execution Error: {e}")
