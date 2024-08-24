from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
filename = 'knn.pkl'
model = pickle.load(open(filename, 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the web form
    fixed_acidity = request.form.get('fixed acidity')
    volatile_acidity = request.form.get('volatile acidity')
    citric_acid = request.form.get('citric acid')
    residual_sugar = request.form.get('residual sugar')
    chlorides = request.form.get('chlorides')
    free_sulfur_dioxide = request.form.get('free sulfur dioxide')
    total_sulfur_dioxide = request.form.get('total sulfur dioxide')
    density = request.form.get('density')
    pH = request.form.get('pH')
    sulphates = request.form.get('sulphates')
    alcohol = request.form.get('alcohol')


    features = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]

    # Convert the list to a 2D array
    features = [list(map(int, features))]

    # Predict the class using the model
    prediction = model.predict(features)[0]


    # Render a new web page with the prediction
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
