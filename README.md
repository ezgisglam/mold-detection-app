# Mold Detection Web Application

This project is a deep learning-based image classification web application designed to detect mold presence in Petri dish images using a convolutional neural network (CNN) developed with PyTorch and deployed via Streamlit.

## Features

- Upload an image and receive a real-time prediction:
  - Clean
  - Mold Detected
  - Invalid (if the image is not suitable)

- Automatic logging of predictions in a CSV file

- Graphical result summaries:
  - Bar chart for prediction counts
  - Pie chart for class distribution
  - Time-based histogram

- Date filtering and message panel for reporting

## Model Information

- Framework: PyTorch
- Model type: Custom CNN (optionally ResNet18)
- Input size: 64x64 pixels
- Classification: Multiclass (Clean, Mold Detected, Invalid)
- Trained model file: `mold_model_optimized.pt`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ezgisglam/mold_detection_app.git
cd mold_detection_app

