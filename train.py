import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from load_data import load_data
from model import countception
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import numpy as np

def main(debug = False, model_name = 'countception_model'):

    if debug:
        if print(tf.test.is_built_with_cuda()):
            print("Training using the GPU!")

        else:
            print("Training using the CPU!")

        print()

    random_seed = 56
    epochs = 15
    batch = 2
    lr = 0.001
    opt = opt = keras.optimizers.Adam(learning_rate=lr)


    x, y = load_data()

    x = np.float32(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, random_state=random_seed
    )

    if debug:
        print('-----raw-----')
        print('X: ', x.shape)
        print('Y: ', y.shape)

        print('-----Parameters-----')
        print('Training: ', x_train.shape)
        print('Test: ', x_test.shape)

        print('-----Labels-----')
        print('Training: ', y_train.shape)
        print('Test: ', y_test.shape)
        print()

    model = countception()
    shape = x_test.shape
    model.build(input_shape=shape)
    
    if debug:
        print('Model Summary:')
        print(model.summary())
        print()

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch)
    model.save('./models/' + model_name)

    print("Model trained!")

    display(x_train[0], y_train[0], model.predict(x_train[0]).reshape((1, 632, 632, 3)))

def display(Original_image, Original_labels, Pred_count):
    plt.imshow(Original_image)
    plt.show()

    plt.imshow(Original_labels)
    plt.show()

    plt.imshow(Pred_count)
    plt.show()


if __name__ == '__main__':
    main()