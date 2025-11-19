# README

## Overview
This repository contains the official implementation accompanying the paper:

**Poison To Detect: Detecting Targeted Overfitting in Federated Learning**  
*[Conference Name, Year]*

The code provides all components necessary to reproduce the experiments, analyses, and figures presented in the paper. It includes data preparation scripts, model training pipelines, evaluation routines, and utilities used during the study.



## Repository Structure
.
├── README.md
├── LICENSE
├── Detection_tools
│   ├── __init__.py
│   ├── datasets.py
│   ├── fingerprinting.py
│   └── utils.py
├── launchers # Directory for launcher scripts using AWS Sagemaker 
│   ├── cifar10.ipynb
│   ├── Pathmnist.ipynb
│   ├── eurosat.ipynb
│   ├── fashionmnist.ipynb
│   └── mnist.ipynb
├── config.py
├── datasets_utils.py
├── loggers
├── main.py
└── models.py
    ├── server.py
    └── utils.py
## 🚀 Quick Start

### Prerequisites

*   Python 3.8+
*   PyTorch 1.9.0+
*   MedMNIST 2.2.2
*   CUDA (optional, for GPU acceleration)

### Installation

1.  **Clone the repository:**
    ```bash    git clone repo
    ```

2.  **(Optional) Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```


### Basic Usage

Run the main training script, example for cifar10 

```bash
python main.py \
  --dataset_name cifar10 \
  --num_clients 20 \
  --frac_clients 1.0 \
  --dirichlet_alpha 0.5 \
  --test_size 0.2 \
  --poison_size 0.35 \
  --rounds 15 \
  --local_epochs 4 \
  --batch_size 64 \
  --client_lr 0.01 \
  --client_momentum 0.9 \
  --weight_decay 1e-4 \
  --server_opt fedopt \
  --server_lr 0.001 \
  --enable_fingerprinting 1 \
  --fingerprint_method sparse \
  --fingerprint_sparsity 0.01 \
  --target_dot_strength 1.0 \
  --honest_fraction 0.1 \
  --detection_margin 1.5 \
  --seed 42 \
  --history_window 5 \
  --method label_flip \
  --label_flip_alpha 1.0 \
  --backdoor_target_label 1 \
  --backdoor_patch_size 15 \
  --backdoor_intensity 1.0 \
  --tau_backdoor_threshold_statistical 0.1 \
  --tau_backdoor_threshold_emprical 0.1 \
  --targeted_clients "1" \
  --verified_clients "0 1 2"
  ```

## License
This project is released under the **MIT License**. See the `LICENSE` file for details.

## Contact
For questions or collaborations, please reach out to:

**[Authors names]**  
Email: **author@example.com**
