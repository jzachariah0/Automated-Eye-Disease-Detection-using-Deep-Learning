import os
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from model import EyeDiseaseDetectionModel

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('saved_models/eye_disease_detection_model.h5')

# Set up allowed extensions for file upload
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
   
