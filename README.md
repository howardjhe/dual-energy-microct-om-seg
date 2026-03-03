# dual-energy-microct-om-seg

This repository contains a sequence of scripts that form a unified integration pipeline for dual energy (33.1 and 33.2 keV) micro-CT OM segmentation processing. The typical workflow uses `tomocupy` for phase retrieval reconstruction, `ANTs` for 3D rigid registration, and a custom `PyTorch` script for subtraction and erosion analysis.

## Installation

The pipeline requires several Python packages to run. You can install the core dependencies using `pip`:

```shell
pip install numpy tifffile antspyx torch torchvision torchaudio scikit-image dipy tqdm tomocupy
```

*Note: For `torch` (PyTorch) and `tomocupy`, you may need specific installation commands depending on your hardware environment (e.g., CUDA for NVIDIA GPUs).*

## Core Scripts

### 1. `tomocupy_process_two_files.py`
**Purpose**: Reconstructs exactly two `.h5` files representing the two energy levels (e.g. 33.1 keV and 33.2 keV) in a target folder using Tomocupy's Paganin phase retrieval.

**Key Arguments:**
- `folder_path`: Target folder containing the `.h5` files.
- `--retrieve-phase-alpha`: Alpha value used for phase retrieval (Default: `0.0001`).
- `--start-slice`, `--end-slice`: Specify the slice range to reconstruct (Default: `0` to `2047`).

### 2. `ants_registration_two_folders.py`
**Purpose**: Performs a 3D rigid registration on a sequence of TIFs between a fixed folder (e.g., 33.1 keV) and a moving folder (e.g., 33.2 keV) using ANTs.

**Key Arguments:**
- `folder_p1`: Path to fixed image sequence (e.g., `33.1_rec`).
- `folder_p2`: Path to moving image sequence (e.g., `33.2_rec`).
- `--start-slice`: Starting slice index for registration stack (Defaults to stack midpoint).
- `--num-slices`: Number of slices to process for a sub-chunk (Default: `300`).
- `--output-dir`: Optional custom output directory.

### 3. `freq_diff_subtraction.py`
**Purpose**: Performs subtraction between the two volumes using frequency difference signal mask (1e3), processes regions of interest (ROI), and executes GPU-accelerated morphological erosion.

**Key Arguments:**
- `--folder1`: Fixed image folder path.
- `--folder2`: Moving image folder path.
- `--transform`: The transformation `.mat` file created in the `ANTs` stage.
- `--roi`: Adjust differences filtering ROI parameter.
- `--low_thresh`: Subtraction difference low threshold limit.
- `--n`, `--iterations`: Kernel shape limit / repetition parameters for erosion.

### 4. `integrate_pipeline.py`
**Purpose**: This master script stitches all three scripts together automatically, so you can execute the entire pipeline with a single command. It evaluates inputs and triggers only the required subsequent parts, passing arguments correctly and logging execution parameters.

**Execution Scenarios & Arguments:**

* **Scenario 1: Start from raw `.h5` files**
  ```shell
  python integrate_pipeline.py --h5-folder /path/to/h5/dir
  ```
  *Pipeline path:* `tomocupy` -> `ANTs` -> `freq_diff_subtraction`

* **Scenario 2: Start from existing image folders**
  ```shell
  python integrate_pipeline.py --folder1 /path/33.1_rec --folder2 /path/33.2_rec
  ```
  *Pipeline path:* `ANTs` (creates transform) -> `freq_diff_subtraction`

* **Scenario 3: Start from image folders and a pre-calculated `.mat` file**
  ```shell
  python integrate_pipeline.py --folder1 /path/33.1_rec --folder2 /path/33.2_rec --transform /path/transform.mat
  ```
  *Pipeline path:* `freq_diff_subtraction`

All standalone optional parameters like `--retrieve-phase-alpha`, `--ants-num-slices`, or `--roi` can be directly provided to the `integrate_pipeline.py` script and will be transparently passed to the lower-level scripts. A log `integration_parameters.txt` is stored when executing parsing decisions.
