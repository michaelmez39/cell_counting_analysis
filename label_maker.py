import tensorflow as tf
from tensorflow.keras import layers
import cv2
import numpy as np
import os
import shutil

def make_labels(debug = False):
    file_path = '../mbm_data'
    dir_name = 'labels'
    dot_path = '../mbm_data/notation'

    cur_dir = os.getcwd()
    label_maker = layers.Conv2D(1, 32, padding='same', use_bias=False, kernel_initializer='Ones')
    ret = []
    notations = []

    dot_list = os.listdir(dot_path)

    for entry in dot_list:
        img = cv2.imread(os.path.join(dot_path, entry))

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

    direct_path = os.path.join(file_path, dir_name)
    if os.path.exists(direct_path):
        shutil.rmtree(direct_path)

    os.mkdir(direct_path)
    
    os.chdir(direct_path)
    for i, img in enumerate(heat):
        img_save = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite('map' + dot_list[i], img_save)

    os.chdir(cur_dir)

if __name__ == '__main__':
    make_labels()