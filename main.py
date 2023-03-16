import argparse
import os

import cv2
import numpy as np
import tensorflow as tf

from model import build_model


def preprocess_image(image):
    # Resize image to expected input size of model
    image = cv2.resize(image, (224, 224))

    # Convert image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to be between 0 and 1
    image = image.astype('float32') / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


def main(image_path, model_path):
    # Load image
    image = cv2.imread(image_path)

    # Preprocess image
    preprocessed_image = preprocess_image(image)

    # Load model
    model = build_model()
    model.load_weights(model_path)

    # Make prediction
    prediction = model.predict(preprocessed_image)

    # Convert prediction to class label
    if prediction > 0.5:
        class_label = 'diseased'
    else:
        class_label = 'healthy'

    print(f"The input image is classified as {class_label}.")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eye Disease Detection Program')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('model_path', type=str, help='Path to trained model')
    args = parser.parse_args()

    main(args.image_path, args.model_path)
