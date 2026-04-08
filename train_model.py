import kagglehub
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Download dataset
path = kagglehub.dataset_download("datamunge/sign-language-mnist")
print("Path to dataset files:", path)

# Use the downloaded path directly for local execution
dataset_path = path

# Load and preprocess data
def load_data():
    train = pd.read_csv(os.path.join(dataset_path, "sign_mnist_train.csv"))
    test = pd.read_csv(os.path.join(dataset_path, "sign_mnist_test.csv"))
    
    X_train = train.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
    y_train = tf.keras.utils.to_categorical(train['label'], num_classes=26)
    
    X_test = test.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
    y_test = tf.keras.utils.to_categorical(test['label'], num_classes=26)
    
    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = load_data()

# Build and train model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(26, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print("Starting model training...")
model = create_model()
history = model.fit(X_train, y_train, epochs=15, validation_split=0.2)

# Save the trained model
model.save('sign_language_model.h5')
print("Model saved successfully as 'sign_language_model.h5'")

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Accuracy: {test_acc:.2%}")

# Optional: Plot training history and confusion matrix
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Progress')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Progress')
plt.legend()
plt.show()