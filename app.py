from flask import Flask, render_template, request
import numpy as np
import os
from utils import *
from models import load_models
# from tensorflow.keras.models import load_model
import keras


cnn_model = keras.models.load_model("saved_models/cnn.h5")
app = Flask(__name__)
models = load_models()

emotion_labels = ["Calm","Happy","Sad","Fear"]

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    path = os.path.join("uploads", file.filename)
    file.save(path)

    data = load_excel(path)

    data = bandpass_filter(data)
    data = notch_filter(data)
    data = normalize(data)

    windows = sliding_window(data)

    predictions = []

    for w in windows:
        w = np.expand_dims(w, axis=0)  # shape (1, 256, 14)

        pred = cnn_model.predict(w, verbose=0)
        label = np.argmax(pred)

        predictions.append(label)

    # Majority voting
    final = max(set(predictions), key=predictions.count)

    result = [emotion_labels[p] for p in predictions]

    return render_template("index.html",
                           result=result,
                           final=emotion_labels[final])

if __name__ == "__main__":
    app.run(debug=True)