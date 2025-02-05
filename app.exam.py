from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("fashion_mnist_cnn.h5")

# Define class labels (for Fashion MNIST)
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Ensure uploads directory exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img = np.array(img) / 255.0  # Normalize to [0,1]
    img = img.reshape(1, 28, 28, 1)  # Reshape for the model
    return img

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        # Save uploaded file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Preprocess and classify image
        img = preprocess_image(file_path)
        predictions = model.predict(img)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence_score = np.max(predictions) * 100  # Convert to percentage

        return render_template("result.html", filename=file.filename, label=predicted_class, confidence=confidence_score)

    return render_template("upload.html")

# Route to display uploaded image and results
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return f'<img src="/static/uploads/{filename}" width="200"><br>'

if __name__ == "__main__":
    app.run(debug=True)
