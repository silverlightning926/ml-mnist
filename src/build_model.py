from keras.api.models import Sequential
from keras.api.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.api.optimizers import RMSprop
from keras.api.losses import categorical_crossentropy
from keras.api.metrics import Accuracy

import os


def _build_model():

    if os.path.exists('./model.keras'):
        return Sequential().load('model.keras')

    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=RMSprop(),
                  loss=categorical_crossentropy,
                  metrics=[Accuracy()])

    return model


def get_model():
    return _build_model()
