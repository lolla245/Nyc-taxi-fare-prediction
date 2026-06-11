import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    # drop missing values
    df = df.dropna()

    # remove invalid coordinates (basic cleaning)
    df = df[
        (df['pickup_longitude'] != 0) &
        (df['pickup_latitude'] != 0) &
        (df['dropoff_longitude'] != 0) &
        (df['dropoff_latitude'] != 0)
    ]

    return df


def feature_engineering(df):
    # example: distance feature (simple Euclidean approx)
    df['distance'] = np.sqrt(
        (df['dropoff_longitude'] - df['pickup_longitude'])**2 +
        (df['dropoff_latitude'] - df['pickup_latitude'])**2
    )

    return df