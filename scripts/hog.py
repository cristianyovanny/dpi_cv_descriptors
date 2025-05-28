import os
import numpy as np
import cv2 as cv
from utils.image_loader import load_images_and_labels

def extract_hog_descriptor(image):
    hog = cv.HOGDescriptor(_winSize=(64, 128),
                           _blockSize=(16, 16),
                           _blockStride=(8, 8),
                           _cellSize=(8, 8),
                           _nbins=9)
    descriptor = hog.compute(image)
    print("Resized shape:", descriptor.shape)
    return descriptor.flatten()  # shape: (1, N)

def extract_and_save_hog_features(query_dir, gallery_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for dir_name, prefix in zip([query_dir, gallery_dir], ['query', 'gallery']):
        images, labels = load_images_and_labels(dir_name)
        X, Y = [], []
        for img, label in zip(images, labels):
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            resized = cv.resize(gray, (64, 128))
            descriptor = extract_hog_descriptor(resized)
            print("Feature shape:", descriptor.shape)
            X.append(descriptor)
            Y.append(label)
        np.save(os.path.join(output_dir, f"{prefix}_feats.npy"), np.vstack(X))
        np.save(os.path.join(output_dir, f"{prefix}_labels.npy"), np.array(Y))
        print(f"[✔] Características HOG ({prefix}) guardadas en: {output_dir}")

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    query_path = os.path.join(BASE_DIR, 'dataset', 'query')
    gallery_path = os.path.join(BASE_DIR, 'dataset', 'bounding_box_test')
    hog_output_path = os.path.join(BASE_DIR, 'features', 'hog')
    extract_and_save_hog_features(query_path, gallery_path, hog_output_path)
