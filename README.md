# TFMix: A Robust Time-Frequency Mixing Approach for Domain Generalization in Specific Emitter Identification

This repository contains the official implementation of the paper **"TFMix: A Robust Time-Frequency Mixing Approach for Domain Generalization in Specific Emitter Identification"**.

## Introduction

Specific Emitter Identification (SEI) faces significant challenges when dealing with signal variations caused by changing environmental conditions and time-varying hardware characteristics. **TFMix** is a novel domain generalization method designed to address these challenges. It leverages a time-frequency mixing strategy to generate diverse training samples, thereby improving the robustness of SEI models against domain shifts.

## Features

- **Time-Frequency Mixing**: A novel data augmentation technique that mixes signals in both time and frequency domains to simulate realistic channel variations.
- **Domain Generalization**: Designed to train models that generalize well to unseen domains (e.g., data collected on different days or under different conditions).
- **Complex-Valued Neural Networks**: Utilizes complex-valued CNNs to effectively process IQ signal data.

## Project Structure

- `TFMix.py`: The main entry point for training and evaluation. It handles the training loop, validation, and testing across different domains.
- `fe.py`: Defines the Feature Extractor (FE) module.
- `cls.py`: Defines the Classifier (FC) module.
- `complexcnn.py`: Implements complex-valued convolutional layers and operations.
- `get_ManySig_CR_unequal.py`: Data loader script responsible for loading and preprocessing signal data from different domains.

## Requirements

The code is implemented in Python using PyTorch. The main dependencies are:

- Python 3.x
- PyTorch
- NumPy
- TensorBoard

You can install the necessary packages using pip:

```bash
pip install torch numpy tensorboard
```

## Usage

To train and evaluate the model, simply run the `TFMix.py` script:

```bash
python TFMix.py
```

The script will:
1.  Iterate through different dates (domains) defined in the code (`1-1`, `1-19`, `14-7`, `18-2`).
2.  For each iteration, it uses one date as the **target (test) domain** and the others as **source (training) domains**.
3.  Train the model using the TFMix strategy.
4.  Evaluate the model on the held-out test domain.
5.  Log training progress and results to TensorBoard and text files.

## Results

Training logs and checkpoints are saved in the `logs/` and `model_weight/` directories, respectively. Final accuracy results are appended to `result/TFMix_CR_Acc.txt`.

## License

This project is distributed under a **Custom Non-Commercial License**.
See the [LICENSE](LICENSE) file for more details.
**Any form of commercial use is prohibited.**

## Citation

If you find this work useful in your research, please consider citing our paper.
