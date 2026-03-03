import argparse
import sys
import os
import time

from tomocupy_process_two_files import process_tomocupy_pair
from ants_registration_two_folders import process_registration_pair

def run_v1_pipeline(folder1, folder2, transform, args):
    """
    Run freq_diff_subtraction.py with the proper arguments
    using python subprocess or direct module import.
    In this case, we prefer subprocess to keep it isolated due to ANTs/PyTorch memory 
    or we can call the main indirectly if we imported it.
    Since we need to pass args that would normally be parsed by argparse, 
    we'll use subprocess.
    """
    import subprocess
    
    cmd = [
        sys.executable, "freq_diff_subtraction.py",
        "--folder1", str(folder1),
        "--folder2", str(folder2),
        "--transform", str(transform)
    ]
    
    # Add optional args if they are provided / overridden
    if hasattr(args, 'roi') and args.roi is not None:
        cmd.extend(["--roi", str(args.roi)])
    if hasattr(args, 'low_thresh') and args.low_thresh is not None:
        cmd.extend(["--low_thresh", str(args.low_thresh)])
    if hasattr(args, 'n') and args.n is not None:
        cmd.extend(["--n", str(args.n)])
    
    print("\n==================================")
    print("Running Transformation Pipeline (FDS mask)")
    print(f"Command: {' '.join(cmd)}")
    print("==================================\n")
    
    try:
        subprocess.run(cmd, check=True)
        print("\nPipeline execution completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Pipeline execution failed: {e}")
        sys.exit(1)

def save_parameters(args, output_dir):
    """
    Saves all used parameters to a text file in the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    params_path = os.path.join(output_dir, "integration_parameters.txt")
    with open(params_path, "w") as f:
        f.write("=== Integration Pipeline Parameters ===\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write("---------------------------------------\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    print(f"Parameters saved to: {params_path}")

def main():
    parser = argparse.ArgumentParser(description="Master Integration Pipeline: Tomocupy -> ANTs -> Transformation (FDS mask)")
    
    # Input options
    parser.add_argument("--h5-folder", type=str, help="Target folder path containing the .h5 files. If provided, starts from tomocupy step.")
    parser.add_argument("--folder1", type=str, help="Fixed image folder path (e.g., 33.1 keV rec folder)")
    parser.add_argument("--folder2", type=str, help="Moving image folder path (e.g., 33.2 keV rec folder)")
    parser.add_argument("--transform", type=str, help="Path to .mat transform file. If not provided, runs ANTs registration to generate it.")
    
    # Optionals for Tomocupy
    parser.add_argument("--retrieve-phase-alpha", type=float, default=0.0001, help="Tomocupy: Phase retrieval alpha (Default: 0.0001)")
    parser.add_argument("--tomocupy-start-slice", type=int, default=0, help="Tomocupy: Start slice (Default: 0)")
    parser.add_argument("--tomocupy-end-slice", type=int, default=2047, help="Tomocupy: End slice (Default: 2047)")
    
    # Optionals for ANTs
    parser.add_argument("--ants-start-slice", type=int, default=None, help="ANTs: Start slice index for registration")
    parser.add_argument("--ants-num-slices", type=int, default=300, help="ANTs: Number of slices to process (Default: 300)")
    
    # Optionals for Transformation Pipeline (FDS mask)
    parser.add_argument("--roi", type=float, help="Pipeline (FDS mask): ROI threshold for difference")
    parser.add_argument("--low_thresh", type=float, help="Pipeline (FDS mask): Low threshold for subtraction difference")
    parser.add_argument("--n", type=int, help="Pipeline (FDS mask): Erosion kernel size")
    
    args = parser.parse_args()

    # Determine which path to take based on inputs provided
    if args.h5_folder:
        # PATH 1: Start from .h5 files
        print("PATH 1: Starting from .h5 files")
        
        # Step 1: Tomocupy
        print("\n--- STEP 1: Tomocupy Reconstruction ---")
        try:
            recon_folders = process_tomocupy_pair(
                folder_path=args.h5_folder,
                retrieve_phase_alpha=args.retrieve_phase_alpha,
                start_slice=args.tomocupy_start_slice,
                end_slice=args.tomocupy_end_slice
            )
        except Exception as e:
            print(f"Tomocupy Step failed: {e}")
            sys.exit(1)
            
        if len(recon_folders) < 2:
            print("Failed to get two reconstruction folders from Tomocupy.")
            sys.exit(1)
            
        # We assume first folder is 33.1 (fixed) and second is 33.2 (moving) based on the original script
        folder1 = next(f for f in recon_folders if "33.1" in f or "33p1" in f)
        folder2 = next(f for f in recon_folders if "33.2" in f or "33p2" in f)
        
        # Step 2: ANTs Registration
        print("\n--- STEP 2: ANTs Registration ---")
        try:
            _, mat_path = process_registration_pair(
                folder_p1=folder1,
                folder_p2=folder2,
                start_slice=args.ants_start_slice,
                num_slices=args.ants_num_slices
            )
        except Exception as e:
            print(f"ANTs Registration Step failed: {e}")
            sys.exit(1)
            
        if not mat_path:
            print("Failed to generate transformation matrix.")
            sys.exit(1)
            
        # Step 3: Transformation Pipeline
        print("\n--- STEP 3: Transformation Pipeline ---")
        run_v1_pipeline(folder1, folder2, mat_path, args)
        
        folder1_path = os.path.normpath(folder1)
        output_dir = os.path.join(os.path.dirname(folder1_path), f"{os.path.basename(folder1_path)}_subtracted")
        save_parameters(args, output_dir)
        
    elif args.folder1 and args.folder2 and not args.transform:
        # PATH 2: Start from folders, need to generate transform
        print("PATH 2: Starting from image folders (Generate Transform)")
        
        # Step 1: ANTs Registration
        print("\n--- STEP 1: ANTs Registration ---")
        try:
            _, mat_path = process_registration_pair(
                folder_p1=args.folder1,
                folder_p2=args.folder2,
                start_slice=args.ants_start_slice,
                num_slices=args.ants_num_slices
            )
        except Exception as e:
            print(f"ANTs Registration Step failed: {e}")
            sys.exit(1)
            
        if not mat_path:
            print("Failed to generate transformation matrix.")
            sys.exit(1)
            
        # Step 2: Transformation Pipeline
        print("\n--- STEP 2: Transformation Pipeline ---")
        run_v1_pipeline(args.folder1, args.folder2, mat_path, args)
        
        folder1_path = os.path.normpath(args.folder1)
        output_dir = os.path.join(os.path.dirname(folder1_path), f"{os.path.basename(folder1_path)}_subtracted")
        save_parameters(args, output_dir)
        
    elif args.folder1 and args.folder2 and args.transform:
        # PATH 3: Start from folders and existing transform
        print("PATH 3: Starting from image folders and existing Transform")
        run_v1_pipeline(args.folder1, args.folder2, args.transform, args)
        
        folder1_path = os.path.normpath(args.folder1)
        output_dir = os.path.join(os.path.dirname(folder1_path), f"{os.path.basename(folder1_path)}_subtracted")
        save_parameters(args, output_dir)
        
    else:
        print("Error: Invalid combination of arguments provided.")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
