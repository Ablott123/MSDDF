# MSDDF: Leveraging Structured Multi-Task Supervision and High-Quality Synthesis for Deepfake Generalization

> **Note: This project is still under submission and the core codes are being prepared. We will release more details and the complete implementation upon paper acceptance. Stay tuned!**

## Introduction

This repository contains the code and resources for our paper:  
**"MSDDF: Leveraging Structured Multi-Task Supervision and High-Quality Synthesis for Deepfake Generalization"**.

Our work aims to advance deepfake detection by introducing a structured multi-task learning framework and a novel, high-quality data synthesis pipeline. MSDDF is designed to enhance generalization and robustness across various manipulation methods.

## Status

- 📝 Paper under review (not yet published).
- 🔧 Core code will be released after acceptance and further improvement.
- 📦 Current repository contains documentation, planned structure, and example configuration files.

## Table of Contents

- [Introduction](#introduction)
- [Status](#status)
- [Table of Contents](#table-of-contents)
- [Method Overview](#method-overview)
- [Dataset Structure](#dataset-structure)
- [Requirements](#requirements)
- [Usage](#usage)

## Method Overview

- **Twin Images Generator:**  
  We introduce a module for generating semantically consistent image pairs with subtle, fine-grained differences, helping the model detect inconspicuous deepfake artifacts more effectively.

- **Multi-Strategy Dynamic Blender:**  
  Our flexible blending module integrates classic blending and a novel dynamic weighted approach to simulate diverse and realistic forgery patterns, greatly enriching training data diversity.

- **Multi-Task Adversarial Supervision:**  
  We design a multi-task adversarial learning framework where the discriminator predicts manipulation attributes (region, type, strength), enabling the model to learn generalizable and manipulation-invariant features.

- **Extensive Validation:**  
  Experiments on public datasets show that MSDDF achieves superior generalization and robustness compared to state-of-the-art methods.

## Dataset Structure

The datasets used in this project should be organized as follows:

```plaintext
datasets
├── lmdb
│   ├── FaceForensics++_lmdb
│   │   ├── data.mdb
│   │   ├── lock.mdb
├── rgb
│   ├── FaceForensics++
│   │   ├── original_sequences
│   │   │   ├── youtube
│   │   │   │   ├── c23
│   │   │   │   │   ├── videos
│   │   │   │   │   │   └── *.mp4
│   │   │   │   │   ├── frames (optional, for processed data)
│   │   │   │   │   │   └── *.png
│   │   │   │   │   ├── masks (optional, for processed data)
│   │   │   │   │   │   └── *.png
│   │   │   │   │   └── landmarks (optional, for processed data)
│   │   │   │   │       └── *.npy
│   │   │   │   └── c40
│   │   │   │   │   ├── videos
│   │   │   │   │   │   └── *.mp4
│   │   │   │   │   ├── frames (optional)
│   │   │   │   │   │   └── *.png
│   │   │   │   │   ├── masks (optional)
│   │   │   │   │   │   └── *.png
│   │   │   │   │   └── landmarks (optional)
│   │   │   │   │       └── *.npy
│   │   │   ├── actors
│   │   │   │   ├── ...
│   │   │   ├── manipulated_sequences
│   │   │   │   ├── Deepfakes
│   │   │   │   │   ├── c23
│   │   │   │   │   │   ├── videos
│   │   │   │   │   │   │   └── *.mp4
│   │   │   │   │   │   ├── frames (optional)
│   │   │   │   │   │   │   └── *.png
│   │   │   │   │   │   ├── masks (optional)
│   │   │   │   │   │   │   └── *.png
│   │   │   │   │   │   └── landmarks (optional)
│   │   │   │   │   │       └── *.npy
│   │   │   │   │   ├── c40
│   │   │   │   │   │   ├── videos
│   │   │   │   │   │   │   └── *.mp4
│   │   │   │   │   │   ├── frames (optional)
│   │   │   │   │   │   │   └── *.png
│   │   │   │   │   │   ├── masks (optional)
│   │   │   │   │   │   │   └── *.png
│   │   │   │   │   │   └── landmarks (optional)
│   │   │   │   │   │       └── *.npy
│   │   │   │   ├── Face2Face
│   │   │   │   │   ├── ...
│   │   │   │   ├── FaceSwap
│   │   │   │   │   ├── ...
│   │   │   │   ├── NeuralTextures
│   │   │   │   │   ├── ...
│   │   │   │   ├── FaceShifter
│   │   │   │   │   ├── ...
│   │   │   │   ├── DeepFakeDetection
│   │   │   │   │   ├── ...
```

*Other datasets follow similar structures.*

**Note:**  
- Optional folders (`frames`, `masks`, `landmarks`) are only present if using the processed data.
- Please refer to [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) for dataset download and preparation details.
  
## Requirements

Clone this repository and install the required packages:
pip install -r requirements.txt


## Usage

The core code will be published after peer review process.  
Currently, you can:
- Refer to [docs/](docs/) for method descriptions and FAQs.
- Download example configuration files from [configs/](configs/).
