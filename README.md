# Dental Caries Detection using CNN

A deep learning project for automated detection of dental caries (cavities) from dental X-ray images using Convolutional Neural Networks.

## Overview

This project builds a binary classifier that distinguishes between teeth with caries and healthy teeth. The model uses data augmentation and regularization techniques to improve generalization on a relatively small dataset.

## Dataset

- **Training set**: 256 images (2 classes)
- **Test set**: 32 images (2 classes)
- **Image size**: 224x224 pixels
- **Source**: [Kaggle Tooth Decay Dataset](https://www.kaggle.com/datasets)

## Approach

### Data Augmentation
- Rotation (±20°)
- Width/height shifts (20%)
- Shear and zoom transformations
- Horizontal flipping

### Model Architecture
- Conv2D (32 filters, 3x3) → ReLU → MaxPooling
- Flatten → Dense (128 units) with L2 regularization
- Dropout (50%)
- Sigmoid output for binary classification

### Training
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Epochs: 10

## Results

- **Test Accuracy**: 88%
- **AUC**: See ROC curve in notebook

## Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Pillow

## Usage

Open `caries-detection.ipynb` in Jupyter Notebook or Kaggle and run all cells.
