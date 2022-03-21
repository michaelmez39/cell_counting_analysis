import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from load_data import CellDataset

class ConvBlock(keras.Model):
  def __init__(self, filter1, filter3):
    super(ConvBlock, self).__init__()
    self.conv1 = layers.Conv2D(filter1, 1, 1, padding="SAME")
    self.conv3 = layers.Conv2D(filter3, 3, 1, padding="SAME")
    self.bn = layers.BatchNormalization()
    self.activation = layers.LeakyReLU()
    self.concat = layers.Concatenate(axis=0)

  def call(self, inputs, training=False):
    x1 = self.conv1(inputs)
    x1 = self.bn(x1, training=training)
    x1 = self.activation(x1)

    x2 = self.conv1(inputs)
    x2 = self.bn(x2, training=training)
    x2 = self.activation(x2)

    x = self.concat([x1, x2])
    return x

# Layer(3, 64, 3, "valid"),
# FilterBlock(64, 16, 16),
# FilterBlock(32, 16, 32),
# Layer(48, 16, 14, "valid"),
# FilterBlock(16, 112, 48),
# FilterBlock(160, 40, 40),
# FilterBlock(80, 32, 96),
# Layer(128, 16, 17, "valid"),
# Layer(16, 64, 1, "valid"),
# Layer(64, 1, 1, "valid")

model = keras.Sequential([
  layers.ZeroPadding2D(padding=(16, 16)),
  layers.Rescaling(1./255),
  layers.Conv2D(64, 3, padding="VALID"),
  layers.BatchNormalization(),
  layers.LeakyReLU(),
  ConvBlock(16, 16),
  ConvBlock(16, 32),
  layers.Conv2D(16, 14, padding="VALID"),
  layers.BatchNormalization(),
  layers.LeakyReLU(),
  ConvBlock(112, 48),
  ConvBlock(40, 40),
  ConvBlock(32, 96),
  layers.Conv2D(16, 17, padding="VALID"),
  layers.BatchNormalization(),
  layers.LeakyReLU(),
  layers.Conv2D(64, 1, padding="VALID"),
  layers.BatchNormalization(),
  layers.LeakyReLU(),
  layers.Conv2D(1, 1, padding="VALID"),
  layers.BatchNormalization(),
  layers.LeakyReLU(),
])

model.compile(
  optimizer=keras.optimizers.RMSprop(),
  loss = keras.losses.MeanSquaredError(),
  metrics=[keras.metrics.Accuracy()]
)

dataset = CellDataset("../Images/MBM")

model.fit(
  dataset.data.batch(1),
  epochs=2,
)

model.save("model0")