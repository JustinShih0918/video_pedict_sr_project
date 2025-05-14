# Video Frame Interpolation and Super-Resolution Project

## Description

This project demonstrates video processing techniques including:
1.  **Optical Flow Estimation:** Calculates motion between video frames.
2.  **Frame Interpolation:** Generates an intermediate frame (frame1) between two input frames (frame0 and frame2) using optical flow and warping.
3.  **Flow Regularization:** Improves the quality of estimated optical flow fields.
4.  **Super-Resolution:** Enhances the resolution of the generated intermediate frame.

The primary script `main_infer.py` orchestrates these processes.


## Prerequisites

*   Python 3.x (tested with Python 3.9)
*   A virtual environment is recommended.

## Setup and Installation

1.  **Clone the repository (if applicable) or ensure you have the project files.**

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv # you don't have to do this line if you have built the env before.
    source venv/bin/activate # do this line directly if you have built the env
    ```
    *(On Windows, use `venv\Scripts\activate`)*

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Prepare Input Data:**
    *   Place your input frames (e.g., `frame0.png`, `frame2.png`, and optionally `frame1.png` for evaluation) in a directory. The script currently expects them in `data/public/`. You can modify the paths in `main_infer.py` if needed.
2. **go to your branch**
    ```bash
    git pull
    git checkout your_branch
    ```
3.  **Run the main script:**
    ```bash
    python main_infer.py
    ```

4.  **Deactivate the virtual environment when done:**
    ```bash
    deactivate
    ```
## for Justin's mac
1.  **Set up**
```
bash gpu_for_mac.sh
```

2.  **Run train**
```
bash train_for_mac.sh
```
3.  **To log out**
```
conda deactivate
deactivate
```

## Output

The script will generate the following files in the `output/` directory:
*   `frame1_pred_regularized.png`: The super-resolved interpolated frame (predicted frame1).

## Project Structure

*   `main_infer.py`: The main script to run the inference pipeline.
*   `requirements.txt`: Lists Python package dependencies.
*   `models/`: Directory containing the core processing modules:
    *   `__init__.py`: Makes `models` a Python package.
    *   `optical_flow.py`: Contains `estimate_flow` for optical flow calculation.
    *   `fusion.py`: Contains `warp_frame` and `fuse_frames` for frame interpolation.
    *   `super_resolution.py`: Contains `upscale` for increasing frame resolution.
    *   `flow_regularization.py`: Contains `regularize_flow_field` for improving flow quality.
*   `utils/`: Directory for utility functions:
    *   `__init__.py`: Makes `utils` a Python package.
    *   `metrics.py`: Contains `compute_psnr_ssim` for evaluating image quality.
*   `data/`: (Assumed) Directory for input data.
    *   `public/`: (Assumed) Sub-directory for specific datasets like `frame0.png`, `frame1.png`, `frame2.png`.
*   `output/`: Directory where processed images are saved.
*   `venv/`: (If created) Python virtual environment directory.

## how you can extend the project
*   add a new folder `module` add put your module into it
*   direct the output of the module into optical_flow.py
*   make it able to do the rest of the work