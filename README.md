# CS Project – Reef Soundscape Classification

This project explores the use of **underwater acoustics** and **deep learning** to classify reef health states from audio recordings.  
It builds a pipeline for data exploration, model training, and lightweight deployment models.

---

## Project Overview

Healthy coral reefs are vibrant acoustic environments, whereas degraded reefs are quieter and less diverse.  
By training models on reef soundscapes, we can automatically classify reef condition and potentially monitor ecosystem health at scale.

This repository contains three main stages:

1. **Exploratory Data Analysis**  
   - `data.ipynb`  
   Explore reef audio datasets and visualize sound representations (e.g., spectrograms).

2. **Baseline Model Training**  
   - `AudioClassification_all.ipynb`  
   Train a deep learning classifier on a large, labeled dataset of reef audio recordings from multiple reef sites.

3. **Lightweight Transfer Learning Model**  
   - `AudioClassification_Final.ipynb`  
   Fine-tune a compact model on 24 hours of data from a single site (healthy vs degraded reef),  
   initializing from the baseline model’s weights for efficient transfer learning.

---

## Requirements

- Linux (Ubuntu 20.04+ recommended) or WSL2 with GPU support  
- [Conda](https://docs.conda.io/en/latest/miniconda.html)  
- NVIDIA GPU drivers + CUDA/cuDNN compatible with TensorFlow  

---

## Setup

Clone this repository:

```bash
git clone https://github.com/Olli365/CS-project.git
cd CS-project
```

### 1. Create Conda environment
```bash
cd setup
cd env_setup
conda env create -f cs_conda_env.yml
```

### 2. Activate environment
```bash
conda activate cs_project
```

### 3. Verify TensorFlow & GPU
```bash
python tf_test.py
```

Expected output includes TensorFlow version and available GPUs.  
If no GPU is detected, check CUDA paths and run:

```bash
export NVIDIA_DIR=$(dirname $(dirname $(python -c "import nvidia.cudnn; print(nvidia.cudnn.__file__)")))
export LD_LIBRARY_PATH=$(echo ${NVIDIA_DIR}/*/lib/ | sed -r 's/\s+/:/g')${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### 4. Install Poetry dependencies
Return to the root directory:

```bash
cd ..
poetry install
```

If Poetry reports lock issues:

```bash
poetry lock
poetry install
```

---
To validate TensorFlow and key dependencies:

```bash
python tf_test.py
```
## Usage

Run notebooks in order:

1. **Data exploration:**  
   ```bash
   jupyter notebook data.ipynb
   ```

2. **Baseline model training:**  
   ```bash
   jupyter notebook AudioClassification_all.ipynb
   ```

3. **Final lightweight model:**  
   ```bash
   jupyter notebook AudioClassification_Final.ipynb
   ```

---



---
