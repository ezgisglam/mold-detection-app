# Mold Detection Web Application

Graduation Project by Ezgi Sağlam  
Istanbul Aydın University – Department of Software Engineering

This project is a deep learning-based image classification web application that detects mold presence in Petri dish images using a convolutional neural network (CNN) developed with PyTorch and deployed via Streamlit.

## Features

- Upload an image and get real-time classification: Clean, Mold Detected, or Invalid
- Automatically logs predictions in a CSV file
- Visualizes predictions with bar chart, pie chart, and histogram
- Date filtering and message panel for reports

## Model Details

- Framework: PyTorch
- Architecture: Custom CNN (optionally ResNet18)
- Input size: 64x64
- Classes: Clean, Mold Detected, Invalid
- Model file: mold_model_optimized.pt

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ezgisglam/mold-detection-app.git
cd mold-detection-app
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## File Structure

```
.
├── app.py                   # Streamlit interface
├── main.py                  # Model training
├── test.py                  # Model testing
├── mold_model_optimized.pt  # Trained model
├── prediction_log.csv       # Logged predictions
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Files excluded from version control
```

## Developer

Ezgi Sağlam  
Istanbul Aydın University  
Graduation Project – 2025  
GitHub: https://github.com/ezgisglam

---

This project is developed for academic and educational purposes.
