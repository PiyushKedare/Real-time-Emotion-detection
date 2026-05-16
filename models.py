from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import numpy as np
# from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten

def train_svm(X, y):
    model = SVC(kernel='rbf', probability=True)
    model.fit(X, y)
    joblib.dump(model, 'saved_models/svm.pkl')

def train_knn(X, y):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    joblib.dump(model, 'saved_models/knn.pkl')

def train_mlp(X, y):
    model = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300)
    model.fit(X, y)
    joblib.dump(model, 'saved_models/mlp.pkl')

def load_models():
    return {
        "svm": joblib.load('saved_models/svm.pkl'),
        "knn": joblib.load('saved_models/knn.pkl'),
        "mlp": joblib.load('saved_models/mlp.pkl')
    }

def build_cnn(input_shape, num_classes=4):
    model = Sequential()

    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model