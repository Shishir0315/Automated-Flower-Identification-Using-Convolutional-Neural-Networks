import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set parameters
IMAGE_SIZE = (128, 128)
MODEL_PATH = 'models/flower_cnn_model.h5'
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load model once at startup for speed
print("Loading model...")
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file not found at {MODEL_PATH}")

def predict_flower(img):
    if img is None:
        return "Please upload an image."
    
    if model is None:
        return "Model failed to load. Please check logs."
    
    # Preprocess image
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

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
    title="🌸 Flower Species Classifier",
    description="Identifying Daisy, Dandelion, Rose, Sunflower, and Tulip using a fine-tuned MobileNetV2 model.",
)

if __name__ == "__main__":
    interface.launch()
