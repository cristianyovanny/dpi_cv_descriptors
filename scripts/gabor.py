import os
import cv2 as cv
import numpy as np
from utils.image_loader import load_images_and_labels

def build_gabor_kernels():
    kernels = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        for sigma in (1, 3):
            for lambd in (np.pi / 4, np.pi / 2):
                kernel = cv.getGaborKernel((21, 21), sigma, theta, lambd, gamma=0.5)
                kernels.append(kernel)
    return kernels

def extract_gabor_features(image, kernels):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    resized = cv.resize(gray, (64, 64))
    resized = resized.reshape(1, -1)
    feats = []
    for k in kernels:
        fimg = cv.filter2D(resized, cv.CV_8UC3, k)
        feats.append(fimg.mean())
        feats.append(fimg.var())
    return feats

def extract_and_save_gabor_features(query_dir, gallery_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    kernels = build_gabor_kernels()

    for dir_name, prefix in zip([query_dir, gallery_dir], ['query', 'gallery']):
        images, labels = load_images_and_labels(dir_name)
        X, Y = [], []
        for img, label in zip(images, labels):
            descriptor = extract_gabor_features(img, kernels)
            X.append(descriptor)
            Y.append(label)
        np.save(os.path.join(output_dir, f"{prefix}_feats.npy"), np.array(X))
        np.save(os.path.join(output_dir, f"{prefix}_labels.npy"), np.array(Y))
        print(f"[✔] Características Gabor ({prefix}) guardadas en: {output_dir}")

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    query_path = os.path.join(BASE_DIR, 'dataset', 'query')
    gallery_path = os.path.join(BASE_DIR, 'dataset', 'bounding_box_test')
    gabor_output_path = os.path.join(BASE_DIR, 'features', 'gabor')
    extract_and_save_gabor_features(query_path, gallery_path, gabor_output_path)
