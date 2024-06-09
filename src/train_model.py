def fit_model(model, x_train, y_train, epochs=20, batch_size=128, verbose=1):
    model.fit(x_train, y_train, epochs=epochs,
              batch_size=batch_size, verbose=verbose)

    return model


def evaluate_model(model, x_test, y_test):
    return model.evaluate(x_test, y_test)


def save_model(model):
    model.save('model.keras')
