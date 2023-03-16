import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

BATCH_SIZE = 32
IMG_SIZE = 224
NUM_CLASSES = 2
EPOCHS = 10
LEARNING_RATE = 0.001
