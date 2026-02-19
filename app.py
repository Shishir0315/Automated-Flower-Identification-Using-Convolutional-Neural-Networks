import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set parameters
IMAGE_SIZE = (128, 128)
MODEL_PATH = r'models/flower_cnn_model.h5'
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def predict_flower(img):
    if img is None:
        return "Please upload an image."
    
    # Preprocess image
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Load model
    if not os.path.exists(MODEL_PATH):
        return f"Model file not found. Please ensure {MODEL_PATH} exists."
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Predict
    predictions = model.predict(img_array)[0]
    
    # Prepare result
    results = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
    return results

# Create Gradio interface
interface = gr.Interface(
    fn=predict_flower,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Flower Species Classifier",
    description="Upload an image of a flower to identify its species (Daisy, Dandelion, Rose, Sunflower, Tulip).",
)

if __name__ == "__main__":
    interface.launch()
