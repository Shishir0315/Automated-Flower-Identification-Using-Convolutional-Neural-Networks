import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import sys

# Set parameters
IMAGE_SIZE = (128, 128)
MODEL_PATH = r'c:\Users\student\Desktop\image classification\models\flower_cnn_model.h5'
# Class names based on directory structure in flowers/
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def predict_flower(img_path):
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    # Load and preprocess image
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create batch axis
    img_array /= 255.0 # Normalize

    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = 100 * np.max(predictions[0])

    print(f"This image most likely belongs to {predicted_class} with a {confidence:.2f} percent confidence.")
    return predicted_class, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py predict.py <path_to_image>")
    else:
        img_path = sys.argv[1]
        predict_flower(img_path)
