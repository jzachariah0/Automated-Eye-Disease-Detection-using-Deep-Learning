import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import BATCH_SIZE, IMG_SIZE

def get_train_data():
    """Load the training data."""
    train_data_gen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2
    )
    train_data = train_data_gen.flow_from_directory(
        os.path.join('data', 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    return train_data

def get_val_data():
    """Load the validation data."""
    val_data_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    val_data = val_data_gen.flow_from_directory(
        os.path.join('data', 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    return val_data

def get_test_data():
    """Load the test data."""
    test_data_gen = ImageDataGenerator(rescale=1. / 255)
    test_data = test_data_gen.flow_from_directory(
        os.path.join('data', 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        class_mode='categorical
