# Probrnn (2022 Reproduction Environment)

This repository contains the code and environment necessary to reproduce a deep learning experiment originally built in 2022 using TensorFlow 1.10 and CUDA 9.2.

All dependencies are packaged in a Docker container for easy setup and long-term reproducibility.

---

## 📦 Project Structure

├── Dockerfile # Defines the full software environment
 ├── requirements.txt # Python package dependencies
 ├── probrnn_env.yml # Original Conda environment (for reference)
 ├── rmodel_loop.py # Wrapper script! Run this to train RNN and edit params.
 ├── model_feedback_varying_rdk_lo_coh.py # Python script specifying RNN structure
 ├── main_feedback_varying_rdk_lo_coh.py # Python script specifying RNN training
 ├── utils.py # Python script specifying RNN structure   
 └── README.md # You're here!


---

## 🚀 Getting Started

### 1. Prerequisites

- **Docker** installed: [Get Docker](https://docs.docker.com/get-docker/)
- **NVIDIA driver ≥ 396.26**
- **NVIDIA Container Toolkit** installed:  
  [Installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Check your setup:
```bash
docker --version
nvidia-smi


### 2. Build the Docker Image
docker build -t probrnn:2022

### 3. Run the Container
docker run --gpus all -it probrnn:2022

### 4. Run the Training Script
python rmodel_loop.py



