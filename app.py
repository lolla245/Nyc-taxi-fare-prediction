from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("taxi_fare_model.pkl")


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        passenger_count = float(request.form['passenger_count'])
        trip_distance = float(request.form['trip_distance'])
        pickup_hour = float(request.form['pickup_hour'])
        pickup_day = float(request.form['pickup_day'])
        store_flag = float(request.form['store_and_fwd_flag'])

        # IMPORTANT: must match training feature order
        features = [
            passenger_count,
            trip_distance,
            pickup_hour,
            pickup_day,
            store_flag
        ]

        final_features = np.array(features).reshape(1, -1)

        # Predict
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return render_template(
            "index.html",
            prediction_text=f"🚖 Estimated Fare: {output}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


# Run app (production + local compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)