import load_dataset
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.api.optimizers import RMSprop
from keras.api.losses import categorical_crossentropy
from keras.api.metrics import Accuracy


def _build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
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
