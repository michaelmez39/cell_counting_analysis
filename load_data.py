import cv2
import numpy as np
import os

def load_data():
    notation_path = '../mbm_data/labels'
    image_path = '../mbm_data/image'
    notations = []
    images = []

    dot_list = os.listdir(notation_path)
    image_list = os.listdir(image_path)

    for entry in dot_list:
        img = cv2.imread(os.path.join(notation_path, entry))
        # Insert any image preprocessing here!
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.float32(img)
        img = img[..., np.newaxis]
        
        
        notations.append(img)

    for entry in image_list:
        img = cv2.imread(os.path.join(image_path, entry))

        # Insert any image preprocessing here! TODO

        images.append(img)

    return np.array(images), np.array(notations)