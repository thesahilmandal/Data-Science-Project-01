from flask import Flask, render_template, request
import os
import numpy as np
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)  # Initializing the Flask app


@app.route("/", methods=["GET"])
def home_page():
    """Render the home page."""
    return render_template("index.html")


@app.route("/train", methods=["GET"])
def train_model():
    """Trigger model training."""
    exit_code = os.system("python main.py")
    if exit_code == 0:
        return "Training Successful!"
    else:
        return "Training Failed!", 500


@app.route("/predict", methods=["POST", "GET"])
def predict():
    """Handle prediction requests."""
    if request.method == "POST":
        try:
            # Read form inputs
            features = [
                float(request.form["fixed_acidity"]),
                float(request.form["volatile_acidity"]),
                float(request.form["citric_acid"]),
                float(request.form["residual_sugar"]),
                float(request.form["chlorides"]),
                float(request.form["free_sulfur_dioxide"]),
                float(request.form["total_sulfur_dioxide"]),
                float(request.form["density"]),
                float(request.form["pH"]),
                float(request.form["sulphates"]),
                float(request.form["alcohol"]),
            ]

            # Convert to NumPy array for model prediction
            data_array = np.array(features).reshape(1, -1)

            # Run prediction
            predictor = PredictionPipeline()
            prediction = predictor.predict(data_array)

            return render_template("results.html", prediction=str(prediction[0]))

        except Exception as e:
            app.logger.error(f"Prediction error: {e}")
            return "An error occurred during prediction.", 500

    # If GET request, show the input form
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
