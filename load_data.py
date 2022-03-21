import tensorflow as tf
import numpy as np
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.data import Dataset
from PIL import Image
img_height = 600
img_width = 600
data_dir = "../Images/MBM"
batch_size = 32


class CellDataset:
  def __init__(self, path):
    self.root_dir = pathlib.Path(path)
    self.images = self.root_dir/"image"
    self.labels = self.root_dir/"label"
    l = [(str(i), str(l)) for i, l in zip(self.images.iterdir(), self.labels.iterdir())]
    ds = Dataset.from_tensor_slices(l)
    self.data = ds.map(self.process_path)
    return

  """
  Turns point label image into density map for training
  """
  def make_label(self, dots):
    #resize = tf.constant([1, 600, 600, 1], dtype=tf.int32)
    #img = tf.reshape(dots, resize)
    img = tf.cast(dots, tf.float32)

    kernel_in = np.ones((32, 32, 1, 1))
    kernel = tf.constant(kernel_in, dtype=tf.float32)
    img = tf.image.resize_with_pad(img, 632, 632)
    img = tf.reshape(img, (1, 632, 632, 1))
    img =  tf.nn.conv2d(img, kernel, 1, padding="VALID")
    img = tf.reshape(img, (601, 601, 1))
    return img

  def process_path(self, paths):
    # resize1 = tf.constant([1, 600, 600, 3])
    # resize2 = tf.constant([1, 601, 601, 1])

    img_raw = tf.io.read_file(paths[0])
    img = tf.image.decode_png(img_raw, channels=3)


    lbl_raw = tf.io.read_file(paths[1])
    lbl = self.make_label(tf.image.decode_png(lbl_raw, channels=1))
    lbl.set_shape([601,601,1])
    return img, lbl


def main():
  ds = CellDataset(data_dir)

  il = ds.data.take(1)

  for img, lbl in il:
    Image.fromarray(img.numpy()).show()
    resize = tf.constant([600, 600])
    lbl = tf.cast(lbl * 20, tf.uint8)
    lbl = tf.reshape(lbl, resize).numpy()
    Image.fromarray(lbl, mode="L").show()

if __name__ == "main":
  main()

