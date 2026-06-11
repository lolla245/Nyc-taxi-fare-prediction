import joblib
import numpy as np

def load_model(path="model/taxi_fare_model.pkl"):
    return joblib.load(path)


def predict_fare(model, input_data):
    """
    input_data = [features in same order as training]
    """
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)

    return prediction[0]