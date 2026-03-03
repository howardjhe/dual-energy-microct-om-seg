import os
import subprocess
import argparse
from pathlib import Path

def process_tomocupy_pair(folder_path, retrieve_phase_alpha=0.0001, start_slice=0, end_slice=2047):
    """
    Process exactly two .h5 files (one for 33.1 keV and one for 33.2 keV) in the given folder.
    Returns a list of the save folder addresses.
    """
    folder_path = Path(folder_path)
    
    # Get all .h5 files in the directory
    h5_files = list(folder_path.glob("*.h5"))
    
    file_33p1 = None
    file_33p2 = None
    
    # Find matching 33.1 and 33.2 keV files
    for f in h5_files:
        if "33p1" in f.name or "33.1" in f.name:
            file_33p1 = f
        elif "33p2" in f.name or "33.2" in f.name:
            file_33p2 = f
            
    if not file_33p1 or not file_33p2:
        found_33p1 = file_33p1.name if file_33p1 else "None"
        found_33p2 = file_33p2.name if file_33p2 else "None"
        raise ValueError(f"Failed to find both 33.1 and 33.2 keV files in {folder_path}.\nFound 33.1: {found_33p1}\nFound 33.2: {found_33p2}")
        
    files_to_process = [
        (file_33p1, "33.1"),
        (file_33p2, "33.2")
    ]
    
    save_folders = []
    
    for file_path, energy_value in files_to_process:
        print(f"\n>>> Processing: {file_path.name}")
        print(f">>> Auto-detected energy: {energy_value} keV")
        print(f">>> Using phase-alpha value: {retrieve_phase_alpha}")

        # Build tomocupy command
        cmd = [
            "tomocupy", "recon_steps",
            "--reconstruction-type", "full",
            "--rotation-axis-auto", "auto",
            "--retrieve-phase-method", "paganin",
            "--retrieve-phase-alpha", str(retrieve_phase_alpha),
            "--energy", str(energy_value),
            "--propagation-distance", "100",
            "--pixel-size", "0.92",
            "--nproj-per-chunk", "64",
            "--fbp-filter", "shepp",
            "--start-row", str(start_slice),
            "--end-row", str(end_slice),
            "--remove-stripe-method", "fw",
            "--fw-sigma", "2.4",
            "--fw-wname", "db30",
            "--fw-level", "6",
            "--file-name", str(file_path),
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Inferred tomocupy save folder path
            # Original logic inference: input=E:\path\to\folder\file.h5 -> output=E:\path\to\folder_rec\file_rec\
            recon_folder_base = folder_path.parent / f"{folder_path.name}_rec"
            save_folder = recon_folder_base / f"{file_path.stem}_rec"
            
            save_folders.append(str(save_folder))
            print(f">>> Expected extraction result saved to: {save_folder}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error processing file {file_path.name}: {e}")
            
    print("\nAll tasks completed!")
    return save_folders

def remove_gibbs_artifacts(recon_folder, slice_start=1260, slice_end=1300, crop_size=1500):
    """
    Utility function: Used to center crop the reconstructed TIFs and remove Gibbs artifacts
    """
    import numpy as np
    import tifffile as tiff
    from dipy.denoise.gibbs import gibbs_removal
    from tqdm import tqdm
    import gc

    recon_folder = Path(recon_folder)
    all_tif_files = sorted([f for f in recon_folder.iterdir() if f.suffix in (".tif", ".tiff")])
    
    tif_files = all_tif_files[slice_start:slice_end]
    
    print(f"\nStarting gibbs_removal, processing slice range: [{slice_start}:{slice_end if slice_end else 'END'}]")
    print(f"Actual exact slices to process: {len(tif_files)} (Total {len(all_tif_files)} slices)")
    
    for tif_path in tqdm(tif_files):
        try:
            data = tiff.imread(str(tif_path)).astype(np.float32)
            h, w = data.shape
            
            start_y = h // 2 - crop_size // 2
            start_x = w // 2 - crop_size // 2
            end_y = start_y + crop_size
            end_x = start_x + crop_size
            
            roi = data[start_y:end_y, start_x:end_x].copy()
            corrected_roi = gibbs_removal(roi, slice_axis=0, n_points=3)
            data[start_y:end_y, start_x:end_x] = corrected_roi
            
            tiff.imwrite(str(tif_path), data.astype(np.float32))
            
            del data, roi, corrected_roi
            gc.collect()
            
        except Exception as e:
            print(f"\nSkipping file {tif_path.name}, Error: {e}")
            
    print("\nGibbs artifact removal for the specified range is completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tomocupy reconstruction, process exactly two .h5 files (33.1 and 33.2 keV).")
    parser.add_argument("folder_path", type=str, help="Target folder path containing the .h5 files")
    parser.add_argument("--retrieve-phase-alpha", type=float, default=0.0001, help="Paganin phase retrieval alpha value (Default: 0.0001)")
    parser.add_argument("--start-slice", type=int, default=0, help="Start slice index for processing (Default: 0)")
    parser.add_argument("--end-slice", type=int, default=2047, help="End slice index for processing (Default: 2047)")
    
    args = parser.parse_args()
    
    try:
        # Run the main logic and get the two returned result folder addresses
        folders = process_tomocupy_pair(
            folder_path=args.folder_path,
            retrieve_phase_alpha=args.retrieve_phase_alpha,
            start_slice=args.start_slice,
            end_slice=args.end_slice
        )
        
        print("\nFinal returned Save Folder addresses:")
        for f in folders:
            print(f)
            
    except Exception as e:
        print(f"Execution failed: {e}")
