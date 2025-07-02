from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import cv2
from skimage.feature import hog

app = Flask(__name__)
model = joblib.load("model_svm.pkl")
scaler = joblib.load("scaler.pkl")

def detect_face(img_path):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    image_color = cv2.imread(img_path)
    if image_color is None:
        return "Invalid image"

    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, 1.1, 5)
    if len(faces) == 0:
        return "No face found"

    x, y, w, h = faces[0]
    face_color = image_color[y:y+h, x:x+w]
    face_gray = image_gray[y:y+h, x:x+w]

    face_resized_gray = cv2.resize(face_gray, (128, 128))
    face_resized_color = cv2.resize(face_color, (64, 64))

    features, _ = hog(face_resized_gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    hist_b = cv2.calcHist([face_resized_color], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([face_resized_color], [1], None, [32], [0, 256]).flatten()
    hist_r = cv2.calcHist([face_resized_color], [2], None, [32], [0, 256]).flatten()

    full_features = np.concatenate([features, hist_b, hist_g, hist_r])
    x = np.array(full_features).reshape(1, -1)
    x_scaled = scaler.transform(x)

    return model.predict(x_scaled)[0]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file"
    file = request.files["image"]
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    result = detect_face(filepath)
    if result == 0:
        name = "Kane Williamson"
    elif result == 1:
        name= "Kobe Bryant"
    elif result == 2:
        name = "Maria Sharapova"
    elif result == 3:
        name = "Cristiano Ronaldo"
    return name

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
