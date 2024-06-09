import load_dataset
import build_model
import train_model


def main():
    (x_train, y_train), (x_test, y_test) = load_dataset.get_mnist()

    model = build_model.get_model()

    model = train_model.fit_model(model, x_train, y_train)

    loss, accuracy = train_model.evaluate_model(model, x_test, y_test)

    print(f'Loss: {loss}, Accuracy: {accuracy}')

    train_model.save_model(model)


if __name__ == '__main__':
    main()
