import os
import argparse
import tensorflow as tf
from model import EyeDiseaseDetectionModel

# Set up argument parser
parser = argparse.ArgumentParser(description='Train Eye Disease Detection Model')
parser.add_argument('--train_dir', type=str, default='data/train', help='Path to training directory')
parser.add_argument('--val_dir', type=str, default='data/val', help='Path to validation directory')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--save_dir', type=str, default='saved_models', help='Path to save trained models')
args = parser.parse_args()

# Load data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    args.train_dir,
    image_size=(224, 224),
    batch_size=args.batch_size,
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='training'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    args.train_dir,
    image_size=(224, 224),
    batch_size=args.batch_size,
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='validation'
)

# Create model
model = EyeDiseaseDetectionModel()

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(train_ds, epochs=args.epochs, validation_data=val_ds)

# Save model
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

model.save(os.path.join(args.save_dir, 'eye_disease_detection_model.h5'))
print('Model saved successfully.')
