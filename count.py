import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


model = keras.Sequential([
  layers.Rescaling(1./255),
  layers.Conv2d(64, 3),
  layers.BatchNormalization(),
  layers.Conv2d(16, 1, activation="selu"),
  layers.Conv2d(16, 3, padding="same", activation="selu"),
  layers.BatchNormalization(),
  layers.Conv2d(16, 1, activation="selu"),
  layers.Conv2d(32, 3, padding="same", activation="selu"),
  layers.BatchNormalization(),
  layers.Conv2d(16, 14, activation="selu"),
  layers.BatchNormalization(),
  layers.Conv2d(112, 1, activation="selu"),
  layers.Conv2d(48, 3, padding="same", activation="selu"),
  layers.BatchNormalization(),
  layers.Conv2d(40, 1),
  layers.Conv2d(40, 3, padding="same", activation="selu"),
  layers.BatchNormalization(),
  layers.Conv2d(32, 1, activation="selu"),
  layers.Conv2d(96, 3, padding="same", activation="selu"),
  layers.BatchNormalization(),
  layers.Conv2d(16, 17, activation="selu"),
  layers.BatchNormalization(),
  layers.Conv2d(64, 1, activation="selu"),
  layers.BatchNormalization(),
  layers.Conv2d(64, 1, activation="selu"),
  layers.BatchNormalization(),
])

model.compile(
  optimizer=keras.optimizers.RMSprop(),
  loss = keras.losses.MeanSquaredError(),
  metrics=[keras.metrics.Accuracy()]
)

model.fit(

)