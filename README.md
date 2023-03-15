# Automated-Eye-Disease-Detection-using-Deep-Learning

This program uses machine learning techniques to detect eye diseases from retinal images. The program is written in Python and uses the Keras library to build and train a deep learning model.

Dataset
The program uses a public dataset of retinal images with labels indicating the presence or absence of various eye diseases, including diabetic retinopathy, glaucoma, and age-related macular degeneration. The dataset is preprocessed to resize the images to a uniform size and normalize pixel values.

Model
The deep learning model used in the program is based on a convolutional neural network (CNN) architecture. Specifically, the model uses a VGG16 network with additional layers added to the top to adapt the model for the specific task of eye disease detection. The model is trained using binary cross-entropy loss and Adam optimization.

Usage
To use the program, simply run the eye_disease_detection.py file and provide an input image. The program will output a prediction indicating the presence or absence of each of the eye diseases in the input image.

Dependencies
The program requires the following dependencies:

Python 3.6 or higher
Keras 2.4.3 or higher
Tensorflow 2.3.0 or higher
NumPy 1.18.5 or higher
Matplotlib 3.2.1 or higher
These dependencies can be installed using the requirements.txt file provided.

References

Dataset: Kaggle Eye Diseases Recognition Dataset: https://www.kaggle.com/monaalshehry/diabeticretinopathy-detection
VGG16 model: Very Deep Convolutional Networks for Large-Scale Image Recognition: https://arxiv.org/abs/1409.1556
