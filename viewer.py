import tensorflow as tf
from PIL import Image
from load_data import CellDataset
import matplotlib.pyplot as plt
import numpy as np

model = tf.keras.models.load_model("model1")
dataset = CellDataset("../Images/MBM").data
print(model.summary())
ds = dataset.shuffle(45).take(1).batch(1)
f, axarr = plt.subplots(1, 3)
def count_cells(input):
    return np.sum(input) / (32 ** 2)

for x, y in ds.as_numpy_iterator():
    print(y.shape)
    y = y.reshape((601, 601))
    x = x.reshape((632, 632, 3))

    axarr[0].imshow(x)
    axarr[1].imshow(y)

    x = x.reshape((1, 632, 632, 3))
    pred = model.predict(x)
    print(pred.shape)
    p = pred[0].reshape((601, 601))
    print("Number of cells", count_cells(y))
    print("Model predicts", count_cells(p))
    axarr[2].imshow(p)

plt.show()
