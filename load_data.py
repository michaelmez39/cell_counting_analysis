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


class CellDataset:
    def __init__(self, data_dir, img_width, img_height):
        self.data_dir = pathlib.Path(data_dir)
        self.img_shape = (img_width, img_height)
        self.images = keras.utils.image_dataset_from_directory(
            str(self.data_dir/'image'), label_mode=None, color_mode="rgb", image_size=self.img_shape)
        self.labels = keras.utils.image_dataset_from_directory(
            str(self.data_dir/'label'),
            label_mode=None,
            color_mode="grayscale",
            image_size=self.img_shape).map(self.make_label
            ).map(layers.Rescaling(1./255))
        self.num_images = self.images.cardinality().numpy()
        num_labels = self.labels.cardinality().numpy()
        assert self.num_images == num_labels
    """
    Turns point label image into density map for training
    """
    def make_label(_, dots):
        kernel_in = np.ones((32, 32, 1, 1))
        kernel = tf.constant(kernel_in, dtype=tf.float32)
        print(dots.shape)
        return tf.nn.conv2d(dots, kernel, strides=[1, 1, 1, 1], padding="VALID")


def main():
    dataset = CellDataset("../Images/MBM", 600, 600)
    # This will display the first image in the dataset as a heatmap
    first = [a for a in dataset.labels.as_numpy_iterator()][0]
    scaled = first * 20
    im = PIL.Image.fromarray(scaled.astype(np.uint8).reshape((32, 569, 569))[1, :,:])
    im.show()


if __name__ == '__main__':
    main()
