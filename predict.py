import argparse
import tensorflow as tf
from model import EyeDiseaseDetectionModel

# Set up argument parser
parser = argparse.ArgumentParser(description='Predict Eye Disease from Image')
parser.add_argument('--image_path', type=str, required=True, help='Path to image file')
parser.add_argument('--model_path', type=str, required=True, help='Path to trained model file')
args = parser.parse_args()

# Load model
model = tf.keras.models.load_model(args.model_path)

# Load image and preprocess
img = tf.keras.preprocessing.image.load_img(args.image_path, target_size=(224, 224))
img_arr = tf.keras.preprocessing.image.img_to_array(img)
img_arr = tf.expand_dims(img_arr, axis=0)
img_arr /= 255.

# Make prediction
prediction = model.predict(img_arr)[0]

# Print result
if prediction[0] > 0.5:
    print('Disease detected.')
else:
    print('No disease detected.')
