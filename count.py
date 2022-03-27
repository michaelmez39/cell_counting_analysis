import tensorflow as tf

# turn off gpu training
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow import keras
from tensorflow.keras import layers
from load_data import CellDataset

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)



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
    x2 = self.conv1(inputs)
    
    x = self.concat([x1, x2])
    x = self.bn(x, training=training)
    x = self.activation(x)

    return x
# diet_model =  keras.Sequential([
#   layers.Rescaling(1./255),
#   layers.Conv2D(24, 3, padding="VALID"),
#   layers.BatchNormalization(),
#   layers.LeakyReLU(),
#   ConvBlock(12, 10),
#   layers.Conv2D(8, 14, padding="VALID"),
#   layers.BatchNormalization(),
#   layers.LeakyReLU(),
#   ConvBlock(8, 12),
#   layers.Conv2D(12, 17, padding="VALID"),
#   layers.BatchNormalization(),
#   layers.LeakyReLU(),
#   ConvBlock(6, 6),
#   ConvBlock(3, 4),
#   layers.Conv2D(8, 1, padding="VALID"),
#   layers.BatchNormalization(),
#   layers.LeakyReLU(),
#   layers.Conv2D(1, 1, padding="VALID"),
#   layers.ReLU(),
# ])

# diet_model =  keras.Sequential([
#   layers.Rescaling(1./255),
#   layers.Conv2D(64, 3, padding="VALID"),
#   layers.BatchNormalization(),
#   layers.LeakyReLU(),
#   ConvBlock(18, 24),
#   layers.Conv2D(32, 14, padding="VALID"),
#   layers.BatchNormalization(),
#   layers.LeakyReLU(),
#   layers.Conv2D(24, 17, padding="VALID"),
#   ConvBlock(18, 24),
#   layers.BatchNormalization(),
#   layers.LeakyReLU(),
#   layers.Conv2D(16, 1, padding="VALID"),
#   layers.BatchNormalization(),
#   layers.LeakyReLU(),
#   layers.Conv2D(1, 1, padding="VALID"),
#   layers.ReLU(),
# ])

# diet_model =  keras.Sequential([
#   layers.Rescaling(1./255),
#   layers.Conv2D(16, 3, padding="VALID"),
#   layers.BatchNormalization(),
#   layers.LeakyReLU(),
  
#   ConvBlock(12, 14),

#   layers.Conv2D(16, 12, padding="VALID"),
#   layers.BatchNormalization(),
#   layers.LeakyReLU(),

#   layers.Conv2D(24, 17, padding="VALID"),
#   layers.BatchNormalization(),
#   layers.LeakyReLU(),

#   ConvBlock(18, 24),
#   layers.Conv2D(1, 64, padding="VALID"),
#   layers.BatchNormalization(),
#   layers.LeakyReLU(),
#   layers.Conv2D(1, 1, padding="VALID"),
#   layers.ReLU(),
# ])

diet_model = keras.Sequential([
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
  layers.LeakyReLU(),
  layers.Conv2D(1, 1, padding="VALID"),
  layers.ReLU(),
])

epochs = 100
batches = 1
lr = 5e-3
number_val = 12

es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

diet_model.compile(
  optimizer=keras.optimizers.Adam(learning_rate=lr),
  loss = keras.losses.MeanSquaredError(),
  metrics=[keras.metrics.Accuracy()],
)

dataset = CellDataset("../Images/adipocyte_data").data.shuffle(40)
test_dataset = dataset.take(number_val).batch(batches)
train_dataset = dataset.skip(number_val).batch(batches)

diet_model.fit(
  train_dataset,
  epochs=epochs,
  verbose=1,
  validation_data=test_dataset,
  # callbacks=[es]
)

diet_model.save("model3")