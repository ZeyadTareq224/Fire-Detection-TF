import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split


# Function to load and preprocess images
def load_and_preprocess_images(folder_path, image_size=(224, 224)):
    images = []
    labels = []
    
    # Load fire images
    fire_folder = os.path.join(folder_path, 'fire')
    for filename in os.listdir(fire_folder):
        img_path = os.path.join(fire_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, image_size)
        img = img / 255.0  # Normalize pixel values
        images.append(img)
        labels.append(1)  # Fire label
    
    # Load non-fire images
    non_fire_folder = os.path.join(folder_path, 'non_fire')
    for filename in os.listdir(non_fire_folder):
        img_path = os.path.join(non_fire_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, image_size)
        img = img / 255.0  # Normalize pixel values
        images.append(img)
        labels.append(0)  # Non-fire label
    
    return np.array(images), np.array(labels)

def get_model():
    # Define input shape
    input_shape = (224, 224, 3)

    # Load pre-trained MobileNetV2 model without the top classification layer
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

    # Freeze base layers (optional)
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom output layer for binary classification
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(1, activation='sigmoid')(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model




# Path to your folder containing fire and non-fire images
folder_path = "fireimages"

# Load and preprocess images
images, labels = load_and_preprocess_images(folder_path)
model = get_model()

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

epochs = 10
batch_size = 32

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    validation_data=(X_val, y_val))

# Evaluate the model on the testing set
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)