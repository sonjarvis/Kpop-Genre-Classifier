import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

DATA_PATH = "CheakMusic/Cheak_Data_Mfcc.json"


def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    z = np.array(data["title"])
    return X, y, z


def predict(model, X, z):
    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    return predicted_index


if __name__ == "__main__":
    model = load_model('models/Mfcc/mfcc_models.h5')
    detail_model = load_model('models/Mfcc/mfcc_detail_models.h5')

    X, y, z = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) # scikit-leran 0.19.1 버전이여야함
    X_test = X_test[..., np.newaxis]

    # predict sample
    # predict(model, X_test[0], y_test[0])
    genres_result = predict(model, X_test[0], z)

    if genres_result == 0 or genres_result == 2 or genres_result == 4:
        detail_genres_result = predict(detail_model, X_test[0], z)

        if detail_genres_result == 0:
            print("Target: {}, Predicted label: {}".format(z, '발라드'))

        elif detail_genres_result == 1:
            print("Target: {}, Predicted label: {}".format(z, '포크'))

        elif detail_genres_result == 2:
            print("Target: {}, Predicted label: {}".format(z, '인디음악'))

    elif genres_result == 1:
        print("Target: {}, Predicted label: {}".format(z, '댄스'))

    elif genres_result == 3:
        print("Target: {}, Predicted label: {}".format(z, '랩/힙합'))

    elif genres_result == 5:
        print("Target: {}, Predicted label: {}".format(z, '트로트'))

