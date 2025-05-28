import os
import numpy as np

def check_npy_shapes(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                try:
                    data = np.load(file_path)
                    print(f"{file_path}: shape = {data.shape}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

if __name__ == '__main__':
    # Cambia aquí la ruta a la carpeta donde tienes tus features
    features_dir = '../features'
    check_npy_shapes(features_dir)