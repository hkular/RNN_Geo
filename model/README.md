# Probrnn (TensorFlow 2.0 version)

This repository contains the code and environment necessary to reproduce a deep learning experiment originally built in 2022 using TensorFlow 1.10. HK has since upgraded the code for Tensorflow 2.0. 

All dependencies are packaged in a Docker container for easy setup and long-term reproducibility.

---

## 📦 Project Structure

├── Dockerfile # Defines the full software environment
 ├── requirements.txt # Python package dependencies
 ├── probrnn_env2.yml # Original Conda environment (for reference)
 ├── runmodel_loop.py # Wrapper script! Run this to train RNN and edit training params.
 ├── model_feedback_v2.py # Python script specifying RNN structure
 ├── main_feedback_v2.py # Python script specifying RNN training
 ├── utils_v2.py # Python script specifying RNN structure   
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
docker build -t probrnn:2025

### 3. Run the Container
docker run --gpus all -it probrnn:2025

### 4. Run the Training Script
python runmodel_loop.py



