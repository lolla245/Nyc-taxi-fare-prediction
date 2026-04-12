from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("taxi_fare_model.pkl")

# Home page (UI)
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

        # IMPORTANT: Order must match training data
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


# Run app
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)