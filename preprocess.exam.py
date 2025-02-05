import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
file_path = "C:/Users/Sheikina/PycharmProjects/pythonProject/pythonProject/archive/fashion-mnist_train"  # Update path if needed
df = pd.read_csv(file_path)

# Separate features and labels
y = df.iloc[:, 0].values  # First column is the label
X = df.iloc[:, 1:].values  # Remaining columns are pixel values

# Reshape X to 28x28 images (grayscale)
X = X.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # Normalize to [0,1]

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=10)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display a sample image
plt.imshow(X_train[0].reshape(28, 28), cmap='gray')
plt.title(f"Label: {np.argmax(y_train[0])}")
plt.show()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Helps prevent overfitting
    Dense(10, activation='softmax')  # 10 classes for classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
