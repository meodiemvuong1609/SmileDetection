import os
import cv2
from const import *
from mtcnn.mtcnn import MTCNN
from pathlib import Path

# PREPROCESS DATA

detector = MTCNN()

# Face detection
def face_detection(image_path: str):
    img = cv2.imread(image_path)
    detected_face = detector.detect_faces(img)
    if not detected_face:
        print("Cannot detect face")
        return None, None, None, None, None

    face_position = detected_face[0].get('box')
    x = face_position[0]
    y = face_position[1]
    w = face_position[2]
    h = face_position[3]
    return x, y, w, h, img


# Preprocess data
def preprocess_data(image_folder, des_image_folder):
    Path(image_folder).mkdir(parents=True, exist_ok=True)
    images_path = [os.path.join(image_folder, file)
                   for file in os.listdir(image_folder)]
    for path in images_path:
        x, y, w, h, img = face_detection(path)
        if not x:
            continue
        img = img[y:y+h, x:x+w]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        new_image_path = os.path.join(des_image_folder, os.path.basename(path))
        Path(new_image_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(new_image_path, img)


if __name__ == "__main__":
    data_path = os.path.join(BASE_DIR, "dataset/smile")
    processed_folder = os.path.join(BASE_DIR, "dataset/smile_processed/")
    preprocess_data(data_path, processed_folder)
