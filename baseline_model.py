from keras.models import Model

from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import Adam

from keras.datasets import cifar10

from sklearn.preprocessing.label import LabelBinarizer
from sklearn.model_selection import train_test_split

import numpy as np


def preprocess_image(x):
    # Convert to Gray scale
    x = np.sum(x/3, axis=3, keepdims=True)
    # Normalize gray scale images
    x = (x - 128) / 128
    return x


def prepocess_label(y):
    return LabelBinarizer().fit_transform(y)


class BaselineModel(object):
    def __init__(self):
        self.__build__()

    def __build__(self):
        input = Input(shape=(32, 32, 1))

        conv1 = Conv2D(8, 5, strides=1, padding='valid', activation='relu')(input)
        pool1 = MaxPooling2D(2, 2, padding='same')(conv1)

        conv2 = Conv2D(16, 5, strides=1, padding='valid', activation='relu')(pool1)
        pool2 = MaxPooling2D(2, 2, padding='same')(conv2)

        conv3 = Conv2D(32, 3, strides=1, padding='valid', activation='relu')(pool2)
        pool3 = MaxPooling2D(2, 2, padding='same')(conv3)

        flattened = Flatten()(pool3)
        fc1 = Dense(120, activation='relu')(flattened)
        fc2 = Dense(84, activation='relu')(fc1)

        output = Dense(10, activation='softmax')(fc2)

        self.model = Model(inputs=[input], outputs=[output])
        self.model.compile(optimizer=Adam(lr=0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit_on_cifar10(self):
        # Load dataset
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42,
                                                              stratify=y_train)

        # pre-process
        X_train = preprocess_image(X_train)
        X_valid = preprocess_image(X_valid)
        X_test = preprocess_image(X_test)

        y_train = prepocess_label(y_train)
        y_valid = prepocess_label(y_valid)
        y_test = prepocess_label(y_test)

        # train and evaluate
        history = self.model.fit(X_train, y_train, batch_size=128, epochs=30, shuffle=True,
                                 validation_data=(X_valid, y_valid), verbose=2)
        results = self.model.evaluate(X_test, y_test, verbose=2)
        print(results)


if __name__ == '__main__':
    model = BaselineModel()
    model.fit_on_cifar10()

    # Baseline: [1.178099395942688, 0.61909999999999998]
