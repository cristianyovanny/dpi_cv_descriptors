import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def load_data(features_path):
    X_train = np.load(os.path.join(features_path, 'gallery_feats.npy'))
    y_train = np.load(os.path.join(features_path, 'gallery_labels.npy'))
    X_test = np.load(os.path.join(features_path, 'query_feats.npy'))
    y_test = np.load(os.path.join(features_path, 'query_labels.npy'))
    return X_train, y_train, X_test, y_test

def encode_labels(y_train, y_test):
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)
    return y_train_enc, y_test_enc, encoder.classes_

def build_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gabor_feat_path = os.path.join(BASE_DIR, 'features', 'gabor')
    X_train, y_train, X_test, y_test = load_data(gabor_feat_path)
    y_train_enc, y_test_enc, class_names = encode_labels(y_train, y_test)

    model = build_model(X_train.shape[1], len(class_names))
    model.fit(X_train, y_train_enc, epochs=10000, batch_size=32, validation_split=0.1)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test_enc, y_pred)
    print(f"[✔] Precisión en el conjunto de prueba: {acc * 100:.2f}%")

    # Crear la carpeta si no existe
    model_dir = os.path.join(BASE_DIR, 'features', 'models')
    os.makedirs(model_dir, exist_ok=True)

    model.save(os.path.join(BASE_DIR, 'features/models', 'gabor_model.keras'))
    print("[✔] Modelo Gabor guardado en 'models/gabor_model.keras'")
