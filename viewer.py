import tensorflow as tf
from PIL import Image
from load_data import CellDataset
import matplotlib.pyplot as plt
import numpy as np

model = tf.keras.models.load_model("model4")
image_width = 150
image_height = 150

dataset = CellDataset("../Images/adipocyte_data").data
print(model.summary())
ds = dataset.shuffle(190).take(1).batch(1)
f, axarr = plt.subplots(1, 3)
def count_cells(input):
    return np.sum(input) / (32 ** 2)

for x, y in ds.as_numpy_iterator():
    print(y.shape)
    y = y.reshape((image_width+1, image_height+1))
    x = x.reshape((image_width+32, image_height+32, 3))

    axarr[0].imshow(x)
    axarr[1].imshow(y)

    x = x.reshape((1, image_height+32, image_height+32, 3))
    pred = model.predict(x)
    print(pred.shape)
    p = pred[0].reshape((image_height+1, image_height+1))
    print("Number of cells", count_cells(y))
    print("Model predicts", count_cells(p)/255)
    axarr[2].imshow(p)

plt.show()
