# 🔥 Calories Burnt Predictor

This interactive Streamlit web application estimates calories burned during exercise. 
It uses a pre-trained machine learning model to provide personalized predictions based on user inputs, featuring a clean, custom-styled interface.

## ✨ Features

* **ML-Powered:** Estimates calories using a neural network model.
* **Intuitive UI:** User-friendly interface built with Streamlit.
* **Custom Design:** Dark theme with a background image and centered layout.
* **Data Preprocessing:** Automatically scales inputs for the model.

## 🛠️ Technology Stack

* **Frontend & App Framework:** Streamlit
* **Machine Learning:** TensorFlow (Keras), scikit-learn
* **Data Handling:** NumPy, h5py, joblib
* **Styling:** HTML/CSS

## 📁 Project Structure

calories_burnt_predictor/
├── static/
│   └── images/
│       ├── bg.jpg        
│       └── logo.jpg           
├── app.py                    
├── calorie_predictor_model.h5 
├── calories.csv               
├── CleanedCalories.csv        
├── dataset.ipynb              
├── .gitattributes             
├── requirements.txt           
└── scaler.pkl                

## 📊 How to Use

1.  Enter your age, gender, height, weight, exercise duration, heart rate, and body temperature.
2.  Click "Predict Calories Burnt" to see the estimated calories.

## Contributing

Contributions are welcome! Fork the repo, create a feature branch, and submit a pull request.
