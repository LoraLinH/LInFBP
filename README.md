# LInFBP: Continuous Filtered Backprojection by Learnable Interpolation Network

This repository contains the official implementation of **LInFBP**, a lightweight and plug-and-play continuous representation framework designed to mitigate discretization and interpolation errors in CT backprojection.

---

## Project Structure

The project is organized as follows:

```text
.
├── Datasets/                # Data management
│   ├── cal_fsim.py          # FSIM calculation
│   └── datasets.py          # Data loading and augmentation logic
│   └── imageProcess.py      # Data pre-processing and post-processing
│   └── utils.py             # Helper functions
├── Model/                   # Core architecture implementations
│   ├── backProjNet.py       # Base backprojection network with nearest interpolation
│   ├── backProjNet_L.py     # Proposed L-LInFBP (Linear-basis variant)
│   ├── backProjNet_F.py     # Proposed F-LInFBP (Fourier-basis variant)
│   ├── backProjNet_linear.py# Standard linear interpolation backprojection
│   ├── backProjNet_cubic.py # Standard cubic interpolation backprojection
│   ├── DICDNet.py           # Original DICDNet implementation
│   ├── DICDNet_L.py         # DICDNet enhanced with L-LInFBP
│   ├── DICDNet_F.py         # DICDNet enhanced with F-LInFBP
│   ├── iRadonMap_Net.py     # Original iRadonMap implementation
│   ├── iRadonMap_Net_L.py   # iRadonMap enhanced with L-LInFBP
│   ├── iRadonMap_Net_F.py   # iRadonMap enhanced with F-LInFBP
│   ├── interpolate.py       # Continuous representation & interpolation logic
│   └── model_fbp_*.py       # End-to-end FBP wrappers for various interpolations
├── Solver/                  # Execution scripts
│   ├── train.py             # Training pipeline
│   ├── test.py              # Evaluation pipeline
│   └── pixelIndexCal.py     # CUDA-based coordinate calculation
├── Utils/                   # Helper functions
│   ├── initFunction.py      # Initialization functions
│   └── initParameter.py     # Hyperparameters
├── eval_memory.py           # Efficiency & Memory benchmark script
└── main.py                  # Entry point
```

## Download Precomputed Indices
For 100-view CT reconstruction, LInFBP requires precomputed coordinate indices to accelerate the backprojection process. You can either download the precomputed file or generate it locally.

Please download the **100-view index file** from the link below and place it in the `Results/` directory:

* **Download Link:** [[Link](https://drive.google.com/file/d/1_X0DSq1SlWrhfL7YFBzhGE6dQ7J_xybM/view?usp=drive_link)]
* **File Name:** `indices_100view.dat`

Alternatively, you can generate the indices directly by running the provided script. This will automatically compute and save `indices_100view.dat` into the `Results/` folder:
```bash
python generate_indices.py
```

## Usage

The execution logic of this project is centralized in `Solver/initParameter.py` and `main.py`. Please follow the steps below to configure your environment.

### 1. Training a Model
To start a new training session:

1.  **Configure Parameters**: Open `Solver/initParameter.py` and set:
    * `net_name`: Set to your target model (e.g., `LInFBP_L`, `DICDNet_L`).
    * `isTrain`: Set to `True`.
    * `reload_model`: Set to `False`.
2.  **Verify Entry Point**: Open `main.py` and ensure the model import corresponds to your `net_name` selection.
3.  **Run**:
    ```bash
    python main.py
    ```

### 2. Evaluation / Testing
To evaluate a pre-trained model and generate **PSNR**, **NMSE**, and **FSIM** metrics:

1.  **Configure Parameters**: Open `Solver/initParameter.py` and set:
    * `net_name`: Must match the architecture of the checkpoint.
    * `isTrain`: Set to `False`.
    * `reload_model`: Set to `True`.
2.  **Run**:
    ```bash
    python main.py
    ```

### 3. Efficiency Benchmarking
To measure **Model Size**, **FLOPs**, **Running Time**, and **Peak Memory** usage:
```bash
python eval_memory.py
