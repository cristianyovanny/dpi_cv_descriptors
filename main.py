import os
import sys
import numpy as np
import runpy

from scripts import hog, gabor
from models import hog_train, gabor_train
from evaluation.evaluate_reid import compute_cmc_map
from evaluation.plot_cmc import plot_cmc_curve

VALID_DESCRIPTORS = ['hog', 'gabor']


def extract_features(descriptor):
    print(f"\n[INFO] Extrayendo características con {descriptor.upper()}...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    query_path = os.path.join(BASE_DIR, 'dataset', 'query')
    gallery_path = os.path.join(BASE_DIR, 'dataset', 'bounding_box_test')
    features_path = os.path.join(BASE_DIR, 'features')

    if descriptor == 'hog':
        hog_output_path = os.path.join(features_path, 'hog')
        hog.extract_and_save_hog_features(query_path, gallery_path, hog_output_path)
    elif descriptor == 'gabor':
        gabor_output_path = os.path.join(features_path, 'gabor')
        gabor.extract_and_save_gabor_features(query_path, gallery_path, gabor_output_path)
    print("[✔] Extracción completada.\n")


def train_model(descriptor):
    print(f"\n[INFO] Entrenando modelo con {descriptor.upper()}...")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    feature_dir = os.path.join(BASE_DIR, 'features', descriptor)
    if not os.path.exists(os.path.join(feature_dir, 'gallery_feats.npy')):
        print(f"[✘] No se encontró el archivo de características en: {feature_dir}. Extrae las características primero.")
        return

    # Ejecutar el script de entrenamiento como módulo para usar su bloque __main__
    if descriptor == 'hog':
        runpy.run_module('models.hog_train', run_name='__main__')
    elif descriptor == 'gabor':
        runpy.run_module('models.gabor_train', run_name='__main__')

    print("[✔] Entrenamiento completado.\n")


def evaluate_model(descriptor, show_plot=False):
    print(f"\n[INFO] Evaluando modelo con {descriptor.upper()}...")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    feat_dir = os.path.join(BASE_DIR, 'features', descriptor)
    try:
        q_feats = np.load(os.path.join(feat_dir, 'query_feats.npy'), allow_pickle=True)
        q_labels = np.load(os.path.join(feat_dir, 'query_labels.npy'))
        g_feats = np.load(os.path.join(feat_dir, 'gallery_feats.npy'), allow_pickle=True)
        g_labels = np.load(os.path.join(feat_dir, 'gallery_labels.npy'))

        q_feats = np.vstack(q_feats)
        g_feats = np.vstack(g_feats)

        cmc, mAP = compute_cmc_map(q_feats, q_labels, g_feats, g_labels)

        print(f"\n--- Resultados para {descriptor.upper()} ---")
        print("CMC top-1:", cmc[0])
        print("CMC top-5:", cmc[4])
        print("CMC top-10:", cmc[9])
        print("mAP:", mAP)

        if show_plot:
            plot_cmc_curve(cmc, descriptor)

        print("[✔] Evaluación completada.\n")
    except FileNotFoundError:
        print(f"[✘] Archivos de características no encontrados para {descriptor.upper()}. Asegúrate de haber extraído las características correctamente.")


def get_descriptor_input():
    descriptor = input("Selecciona el descriptor ('hog' o 'gabor'): ").lower()
    if descriptor not in VALID_DESCRIPTORS:
        print("[✘] Descriptor no válido. Intenta de nuevo.")
        return None
    return descriptor


def menu():
    options = {
        '1': 'Extraer características',
        '2': 'Entrenar modelo',
        '3': 'Evaluar modelo',
        '4': 'Evaluar y mostrar curva CMC',
        '5': 'Salir'
    }

    while True:
        print("\n========= MENÚ PRINCIPAL =========")
        for key, val in options.items():
            print(f"{key}. {val}")
        choice = input("Selecciona una opción (1-5): ").strip()

        if choice in ['1', '2', '3', '4']:
            descriptor = get_descriptor_input()
            if descriptor is None:
                continue

        if choice == '1':
            extract_features(descriptor)
        elif choice == '2':
            train_model(descriptor)
        elif choice == '3':
            evaluate_model(descriptor, show_plot=False)
        elif choice == '4':
            evaluate_model(descriptor, show_plot=True)
        elif choice == '5':
            print("¡Hasta luego!")
            break
        else:
            print("[✘] Opción no válida. Intenta de nuevo.")

if __name__ == '__main__':
    menu()

