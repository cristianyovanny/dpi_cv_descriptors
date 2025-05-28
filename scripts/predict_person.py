import os
import sys
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from tensorflow.keras.models import load_model
from scripts.hog import extract_hog_descriptor
from scripts.gabor import build_gabor_kernels, extract_gabor_features


def load_query_image(image_path, descriptor):
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    if descriptor == 'hog':
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        resized = cv.resize(gray, (64, 128))
        feat = extract_hog_descriptor(resized)    # devuelve array 1D de largo 3780
        feature = feat.reshape(1, -1)             # ahora shape (1, 3780)

    elif descriptor == 'gabor':
        kernels = build_gabor_kernels()
        feat = extract_gabor_features(img, kernels)  # lista de 32 valores
        feature = np.array(feat).reshape(1, -1)       # ahora shape (1, 32)
    else:
        raise ValueError("Descriptor no soportado: usa 'hog' o 'gabor'")

    return feature

def is_person(descriptor, image_path, threshold=0.7):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'features/models', f'{descriptor}_model.keras')

    # Carga modelo binario
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo de detección de persona no encontrado en: {model_path}")

    model = load_model(model_path)
    features = load_query_image(image_path, descriptor)
    prob = model.predict(features)[0][0]  # devuelve una probabilidad
    print(f"[INFO] Confianza de que es persona: {prob:.2f}")

    return prob >= threshold

def predict_identity(descriptor, image_path):
    if not is_person(descriptor, image_path):
        print("[✘] Imagen no corresponde a una persona.")
        return "No Persona"

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # (mantenemos las rutas igual)
    gallery_feats_path  = os.path.join(BASE_DIR, 'features', descriptor, 'gallery_feats.npy')
    gallery_labels_path = os.path.join(BASE_DIR, 'features', descriptor, 'gallery_labels.npy')

    # extraemos descriptor de la query
    print(f"[INFO] Extrayendo características de la imagen de consulta...")
    query_feat = load_query_image(image_path, descriptor)   # shape (1,3780) o (1,32)

    # cargamos la galería
    print(f"[INFO] Cargando características de la galería...")
    gallery_feats = np.load(gallery_feats_path, allow_pickle=True)
    gallery_labels = np.load(gallery_labels_path)

    if gallery_feats.ndim == 1:
        gallery_feats = np.vstack(gallery_feats)

    # **¡¡ AQUÍ YA NO USAMOS model.predict !!**
    print(f"[INFO] Calculando similitudes (distancias euclidianas)…")
    # query_feat ya es (1,D); hacemos broadcast restando
    dists = np.linalg.norm(gallery_feats - query_feat, axis=1)
    min_idx = np.argmin(dists)
    predicted_id = gallery_labels[min_idx]

    print(f"[✔] Persona más similar encontrada con ID: {predicted_id}")
    return predicted_id

def select_image_gui():
    root = tk.Tk()
    root.withdraw()

    descriptor = simpledialog.askstring("Descriptor", "Ingresa el descriptor [hog|gabor]:")
    if descriptor is None or descriptor.lower() not in ['hog', 'gabor']:
        messagebox.showerror("Error", "Descriptor no válido. Usa 'hog' o 'gabor'.")
        return

    image_path = filedialog.askopenfilename(title="Selecciona una imagen",
                                            filetypes=[("Imágenes", "*.jpg *.jpeg *.png")])
    if not image_path:
        messagebox.showerror("Error", "No se seleccionó ninguna imagen.")
        return

    try:
        predicted_id = predict_identity(descriptor.lower(), image_path)
        if predicted_id:
            messagebox.showinfo("Resultado", f"Persona identificada: {predicted_id}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


if __name__ == '__main__':
    if len(sys.argv) == 3:
        descriptor = sys.argv[1].lower()
        image_path = sys.argv[2]

        if descriptor not in ['hog', 'gabor']:
            print("[✘] Descriptor no válido. Usa 'hog' o 'gabor'.")
            sys.exit(1)

        try:
            predict_identity(descriptor, image_path)
        except Exception as e:
            print(f"[✘] Error durante la predicción: {e}")
            sys.exit(1)

    else:
        print("[INFO] No se ingresaron argumentos. Abriendo interfaz gráfica...")
        select_image_gui()
