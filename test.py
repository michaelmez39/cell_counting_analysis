import tensorflow as tf
print("Built with cuda", tf.test.is_built_with_cuda())
print("Using cuda", tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices('GPU')
print(gpus)