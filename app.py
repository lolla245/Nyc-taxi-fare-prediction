from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "taxi_fare_model.pkl")

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        passenger_count = float(request.form['passenger_count'])
        trip_distance = float(request.form['trip_distance'])
        pickup_hour = float(request.form['pickup_hour'])
        pickup_day = float(request.form['pickup_day'])
        store_flag = float(request.form['store_and_fwd_flag'])

        features = np.array([
            passenger_count,
            trip_distance,
            pickup_hour,
            pickup_day,
            store_flag
        ]).reshape(1, -1)

        prediction = model.predict(features)[0]
        output = round(float(prediction), 2)

        return render_template(
            "index.html",
            prediction_text=f"🚖 Estimated Fare: {output}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)