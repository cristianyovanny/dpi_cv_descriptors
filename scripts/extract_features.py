import os
import sys
from scripts import hog, gabor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
query_path = os.path.join(BASE_DIR, 'dataset', 'query')
gallery_path = os.path.join(BASE_DIR, 'dataset', 'bounding_box_test')
features_path = os.path.join(BASE_DIR, 'features')

def main():
    descriptor = sys.argv[1] if len(sys.argv) > 1 else 'gabor'

    if descriptor == 'hog':
        hog_output_path = os.path.join(features_path, 'hog')
        hog.extract_and_save_hog_features(query_path, gallery_path, hog_output_path)
    elif descriptor == 'gabor':
        gabor_output_path = os.path.join(features_path, 'gabor')
        gabor.extract_and_save_gabor_features(query_path, gallery_path, gabor_output_path)
    else:
        print(f"[ERROR] Descriptor no soportado: {descriptor}")
        print("Uso: python extract-features.py [hog|gabor]")

if __name__ == '__main__':
    main()
