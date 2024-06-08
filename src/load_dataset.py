from keras.api.datasets import mnist
import keras.api.utils as utils


def _load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)


def _preprocess_images(train_images, test_images):
    _normalize_images(train_images, test_images)
    _reshape_images(train_images, test_images)


def _normalize_images(train_images, test_images):
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255


def _reshape_images(train_images, test_images):
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))


def _encode_labels(train_labels, test_labels):
    train_labels = utils.to_categorical(train_labels)
    test_labels = utils.to_categorical(test_labels)


def get_mnist():
    (train_images, train_labels), (test_images, test_labels) = _load_data()
    _preprocess_images(train_images, test_images)
    _encode_labels(train_labels, test_labels)
    return (train_images, train_labels), (test_images, test_labels)
