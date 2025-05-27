from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model("calorie_predictor_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            data = [
                float(request.form['Age']),
                float(request.form['Gender']),   
                float(request.form['Height']),
                float(request.form['Weight']),
                float(request.form['Duration']),
                float(request.form['Heart_Rate']),
                float(request.form['Body_Temp']),
            ]

            print("Input data for scaling:", data)
            print("Number of features:", len(data))
            print("Scaler expects:", scaler.n_features_in_)

            scaled_data = scaler.transform([data])
            result = model.predict(scaled_data)
            prediction = round(float(result[0][0]), 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
