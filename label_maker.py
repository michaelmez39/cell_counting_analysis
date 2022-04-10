import tensorflow as tf
from tensorflow.keras import layers
import cv2
import numpy as np
import os
import shutil

# Assumes that you have at least two folders in the path
# image = the folder with all of the raw images
# notation = the folder with all of the notated images

# Function will create a third folder called labels that will populate with the labels 
# necessary for fitting the model

def make_labels(debug = False, path = '../Data/research_data'):
    image_folder = 'image'
    notation_folder = 'notation'
    output_folder = 'labels'

    cur_dir = os.getcwd()
    label_maker = layers.Conv2D(1, 32, padding='same', use_bias=False, kernel_initializer='Ones')
    ret = []
    notations = []

    dot_list = os.listdir(os.path.join(path, notation_folder))

    for entry in dot_list:
        img = cv2.imread(os.path.join(path, notation_folder, entry))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
        img = np.float32(img)
        img = img[..., np.newaxis]
        notations.append(img)

    notations = np.array(notations)
    pads = tf.pad(notations, [[0, 0], [16, 16], [16, 16], [0, 0]], 'constant')
    heat = label_maker(pads)
    heat = heat.numpy()

    if debug:
        return heat

    direct_path = os.path.join(path, output_folder)
    if os.path.exists(direct_path):
        shutil.rmtree(direct_path)

    os.mkdir(direct_path)
    
    os.chdir(direct_path)
    for i, img in enumerate(heat):
        cv2.imwrite(dot_list[i], img)

    os.chdir(cur_dir)

if __name__ == '__main__':
    make_labels()