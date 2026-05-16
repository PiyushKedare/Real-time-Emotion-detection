import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from utils import *
from models import *
# from tensorflow import keras

import keras
DATA_PATH = "dataset"

emotion_map = {
    "calm":0,
    "happy":1,
    "sad":2,
    "fear":3
}

emotion_labels = ["calm", "happy", "sad", "fear"]


def get_label_from_filename(file_name):
    base_name = os.path.splitext(file_name)[0].lower()

    if base_name in emotion_map:
        return emotion_map[base_name]

    for emotion, label in emotion_map.items():
        if base_name.startswith(emotion):
            return label

    raise ValueError(f"Unknown emotion file name: {file_name}")

X_de, X_psd, X_dwt, y = [], [], [], []

for file in os.listdir(DATA_PATH):
    label = get_label_from_filename(file)
    data = load_excel(os.path.join(DATA_PATH, file))

    data = bandpass_filter(data)
    data = notch_filter(data)
    data = normalize(data)

    windows = sliding_window(data)

    for w in windows:
        X_de.append(compute_de(w))
        X_psd.append(compute_psd(w))
        X_dwt.append(compute_dwt(w))
        y.append(label)

X_de = np.array(X_de)
X_psd = np.array(X_psd)
X_dwt = np.array(X_dwt)
y = np.array(y)

from keras.utils import to_categorical
from models import build_cnn

# Prepare CNN data
X_cnn = []
y_cnn = []

for file in os.listdir(DATA_PATH):
    label = get_label_from_filename(file)
    data = load_excel(os.path.join(DATA_PATH, file))

    data = bandpass_filter(data)
    data = notch_filter(data)
    data = normalize(data)

    windows = sliding_window(data)

    for w in windows:
        X_cnn.append(w)
        y_cnn.append(label)

X_cnn = np.array(X_cnn)
y_cnn = to_categorical(np.array(y_cnn), num_classes=4)

sample_indices = np.arange(len(y))
train_idx, test_idx = train_test_split(
    sample_indices,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_de_train, X_de_test = X_de[train_idx], X_de[test_idx]
X_psd_train, X_psd_test = X_psd[train_idx], X_psd[test_idx]
X_dwt_train, X_dwt_test = X_dwt[train_idx], X_dwt[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

X_cnn_train, X_cnn_test = X_cnn[train_idx], X_cnn[test_idx]
y_cnn_train, y_cnn_test = y_cnn[train_idx], y_cnn[test_idx]


def summarize_predictions(model_name, y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(emotion_labels))))

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }, pd.DataFrame(cm, index=emotion_labels, columns=emotion_labels)


train_svm(X_de_train, y_train)
train_knn(X_psd_train, y_train)
train_mlp(X_dwt_train, y_train)

trained_models = load_models()

comparison_rows = []
confusion_matrices = {}

svm_pred = trained_models["svm"].predict(X_de_test)
row, cm_df = summarize_predictions("svm", y_test, svm_pred)
comparison_rows.append(row)
confusion_matrices["svm"] = cm_df

knn_pred = trained_models["knn"].predict(X_psd_test)
row, cm_df = summarize_predictions("knn", y_test, knn_pred)
comparison_rows.append(row)
confusion_matrices["knn"] = cm_df

mlp_pred = trained_models["mlp"].predict(X_dwt_test)
row, cm_df = summarize_predictions("mlp", y_test, mlp_pred)
comparison_rows.append(row)
confusion_matrices["mlp"] = cm_df

print("Training Done!")

# Build model
cnn = build_cnn((X_cnn_train.shape[1], X_cnn_train.shape[2]))

# Train
cnn.fit(X_cnn_train, y_cnn_train, epochs=20, batch_size=32, validation_data=(X_cnn_test, y_cnn_test))

cnn_pred = np.argmax(cnn.predict(X_cnn_test, verbose=0), axis=1)
row, cm_df = summarize_predictions("cnn", y_test, cnn_pred)
comparison_rows.append(row)
confusion_matrices["cnn"] = cm_df

# Save
cnn.save("saved_models/cnn.h5")

print("CNN Training Done!")

comparison_df = pd.DataFrame(comparison_rows).set_index("model")
comparison_df = comparison_df[["accuracy", "precision", "recall", "f1_score"]]

print("\nModel Comparison Matrix:")
print(comparison_df.to_string())

comparison_df.to_csv("saved_models/model_comparison.csv")

for model_name, matrix_df in confusion_matrices.items():
    matrix_df.to_csv(f"saved_models/{model_name}_confusion_matrix.csv")
    print(f"\n{model_name.upper()} Confusion Matrix:")
    print(matrix_df.to_string())