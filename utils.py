from utils import load_data, preprocess_image

# import necessary libraries
import tensorflow as tf
import numpy as np

# load the data
train_data, train_labels, test_data, test_labels = load_data()

# preprocess the images
train_data = preprocess_image(train_data)
test_data = preprocess_image(test_data)

# define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

# save the model
model.save('eye_disease_detection_model.h5')
