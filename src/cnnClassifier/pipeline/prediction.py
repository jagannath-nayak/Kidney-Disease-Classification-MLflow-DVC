import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import json

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
    
    def predict(self):
        # Load the trained model
        model = load_model(os.path.join("model", "model.h5"))

        # Load class index mapping
        try:
            with open(os.path.join("model", "class_indices.json"), "r") as f:
                class_indices = json.load(f)
            idx_to_label = {v: k for k, v in class_indices.items()}
        except Exception as e:
            # Fallback (adjust if you know the actual mapping)
            print("Warning: Could not load class_indices.json. Using default.")
            idx_to_label = {0: "Tumor", 1: "Normal"}

        # Load and preprocess the image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0  # Must match training preprocessing
        test_image = np.expand_dims(test_image, axis=0)

        # Predict
        preds = model.predict(test_image)
        
        # For softmax output (2 units)
        if preds.shape[1] == 2:
            predicted_index = np.argmax(preds, axis=1)[0]
            confidence = preds[0][predicted_index]
        # For sigmoid output (1 unit)
        else:
            predicted_index = int(preds[0][0] > 0.5)
            confidence = preds[0][0] if predicted_index == 1 else 1 - preds[0][0]

        label = idx_to_label.get(predicted_index, "Unknown")

        print(f"Prediction: {label} (Confidence: {confidence:.2f})")
        return [{"image": label}]
