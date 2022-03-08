import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import PIL
import pathlib

img_height = 600
img_width = 600
data_dir = pathlib.Path("../Images/MBM")
batch_size = 32

def make_label(dots):
  kernel_in = np.ones((32, 32, 1, 1))
  kernel = tf.constant(kernel_in, dtype=tf.float32) 
  return tf.nn.conv2d(dots, kernel, strides=[1,1,1,1], padding="VALID", data_format="NHWC")

# list_ds_image = tf.data.Dataset.list_files(str(data_dir/'image/*'))
# list_ds_label = tf.data.Dataset.list_files(str(data_dir/'label/*'))
# length_image = list_ds_image.cardinality().numpy()
# length_label = list_ds_label.cardinality().numpy()
# print(f"Labels: {length_label}\nImages:{length_image}")
# assert length_image == length_label

# list_ds_label = list_ds_label.map(make_label)

images = (data_dir/'label').iterdir()
print(images)
im = PIL.Image.open(str(list(images)[0]))
filtered = make_label(np.array(im.getdata()).reshape((1, 600, 600, 1)))
scaled = np.array(filtered).reshape(569, 569) * 20
im = PIL.Image.fromarray(scaled.astype(np.uint8))
im.show()

"""
Turns point label image into density map for training
"""

