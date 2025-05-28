import os
import cv2 as cv

def load_images_and_labels(folder):
    images, labels = [], []
    files = sorted(f for f in os.listdir(folder) if f.endswith('.jpg') or f.endswith('.png'))
    for f in files:
        path = os.path.join(folder, f)
        img = cv.imread(path)
        if img is not None:
            images.append(img)
            label = int(f.split('_')[0])  # asume formato ID_XXX.jpg
            labels.append(label)
    return images, labels
