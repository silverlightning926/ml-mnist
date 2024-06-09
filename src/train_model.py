from keras.api.models import Sequential


def fit_model(model: Sequential, x_train, y_train, epochs=20, batch_size=128, validation_split=0.1, verbose=1):
    model.fit(x_train, y_train, epochs=epochs,
              batch_size=batch_size, validation_split=validation_split, verbose=verbose)

    return model


def evaluate_model(model: Sequential, x_test, y_test):
    return model.evaluate(x_test, y_test)


def save_model(model: Sequential):
    model.save('model.keras')
