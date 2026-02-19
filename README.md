# Flower Species Classification Project

This project uses a Deep Learning model (Transfer Learning with MobileNetV2) to classify images of flowers into five categories: Daisy, Dandelion, Rose, Sunflower, and Tulip.

## Files in this project:
- `train_model.py`: Script to train the model using Transfer Learning.
- `predict.py`: CLI script to run prediction on a single image.
- `app.py`: Web-based interface using Gradio for interactive predictions.
- `requirements.txt`: List of required Python libraries.
- `models/flower_cnn_model.h5`: The trained model file.
- `training_results.png`: Training accuracy and loss plots.

## How to Run:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
The model is already pre-trained, but you can retrain it:
```bash
py train_model.py
```

### 3. Run Web Application
To start the interactive web interface:
```bash
py app.py
```

### 4. Run CLI Prediction
```bash
py predict.py path/to/image.jpg
```
