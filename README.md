<div align="center">

# IJCNN PARKINNG SPOT OCCUPANCY RECOGNITION

This is the repository containing the source code and documentation for the IJCNN 2026 paper Toward Parking Spot Occupancy Recognition: A Self-supervised Approach.

</div>

## 1. Overview

The proposed training pipeline follows a three-stage approach.

### 1.1 Stage 1: Self-supervised pretraining with SimCLR on generic data (ImageNet)

The first stage (1) involves a self-supervised learning phase using the SimCLR framework on generic data (ImageNet). We initialize the weights from the public SimCLR repository ![SimCLR Repository](https://github.com/google-research/simclr) and convert it to Pytorch using the SimCLR converter repository ![SimCLR Converter Repository](https://github.com/tonylins/simclr-converter)

### 1.2 Stage 2: Self-supervised fine-tuning on parking spot occupancy data (PKLot, CNRPark-EXT, PLds)

The second stage (2) consists of fine-tuning the pretrained SimCLR model on parking spot occupancy data using the PKLot, CNRPark-EXT, and PLds datasets. This stage allows the model to adapt to the specific characteristics of parking spot images. This phase is implemented in the `SimCLR` folder.

### 1.3 Stage 3: Supervised fine-tuning on parking spot occupancy data (PKLot, CNRPark-EXT, PLds)

The third stage (3) involves supervised fine-tuning of the model on the same parking spot occupancy datasets. This stage refines the model's ability to classify parking spot occupancy accurately. This phase is implemented in the `Supervised` folder.

This repository contains:
- `SimCLR`: Contains the code for the stage (2).
- `Supervised`: Contains the code for the stage (3).
- `tools`: Contains utility scripts for data processing and evaluation on edge devices.
