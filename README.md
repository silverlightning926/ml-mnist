# Machine Learning Model For Recognizing Handwritten Numbers
*<div style="color:gray;margin-top:-10px;">By Siddharth Rao</div>*

![Python Lint - autopep8 Workflow](https://github.com/silverlightning926/tensorflow-mnist/actions/workflows/python-lint.yaml/badge.svg)

## Table Of Contents
- [Machine Learning Model For Recognizing Handwritten Numbers](#machine-learning-model-for-recognizing-handwritten-numbers)
  - [Summary](#summary)
  - [Current Status](#current-status)
  - [Getting Started](#getting-started)
  - [Dependencies](#dependencies)
  - [Linting with Autopep8](#linting-with-autopep8)
  - [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
  - [Liscense](#liscense)

## Summary
This repo contains my code for training an ML model using Google's TensorFlow library to recognize handwritten numbers. It has been trained and tested on the famous MNIST *(Modified National Institute Of Standards And Technology)* Dataset.

## Current Status
⚠️ ***This Codebase Is Currently A Work In Progress*** ⚠️

## Getting Started

To get started with this machine learning model for recognizing handwritten numbers, follow these steps:

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/your-username/tensorflow-mnist.git
    ```

2. Install the required dependencies by running the following command:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the MNIST dataset by executing the following script:
    ```bash
    python download_dataset.py
    ```

4. Train the model by running the training script:
    ```bash
    python train_model.py
    ```

5. Once the model is trained, you can test it by running the testing script:
    ```bash
    python test_model.py
    ```

6. Congratulations! You have successfully set up and trained the machine learning model for recognizing handwritten numbers using TensorFlow.

Feel free to explore the code and make any modifications as needed.

## Dependencies
- Python3 - Developed on Python Version 3.12.3
- Keras - Developed Keras Version 3.3.3
These requirements can be found in and downloaded by using [requirements.txt](./requirements.txt)

## Linting with Autopep8
To ensure consistent code formatting, you can use Autopep8, a Python library that automatically formats your code according to the PEP 8 style guide. To install Autopep8, run the following command:
```bash
pip install autopep8
```

Once installed, you can use Autopep8 to automatically format your code by running the following command:
```bash
autopep8 --in-place --recursive ./src
```

This will recursively format all Python files in the current directory and its subdirectories.

Remember to run Autopep8 regularly to maintain a clean and consistent codebase.

If you are using VSCode, you can download and the [Autopep8 VSCode Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.autopep8) and add these lines to your `settings.json` to format with Autopep8 automatically as you type and when you save.
```json
"[python]": {
        "editor.formatOnType": true,
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "ms-python.autopep8"
    }
```

## Dataset
The machine learning model for recognizing handwritten numbers in this repository is trained and tested on the MNIST dataset. MNIST, short for *Modified National Institute of Standards and Technology*, is a widely used dataset in the machine learning community for handwritten digit classification tasks. It consists of a training set of 60,000 examples and a test set of 10,000 examples, where each example is a 28x28 grayscale image of a handwritten digit (0 through 9). The dataset is preprocessed and formatted to facilitate training and evaluation of machine learning models.

You can download the MNIST dataset directly from the [MNIST website](http://yann.lecun.com/exdb/mnist/) or through the provided script in this repository.

## Model Architecture
*Coming Soon*

## Liscense
This repository is governed under the MIT license. The repository's license can be found here: [LICENSE](./LICENSE).