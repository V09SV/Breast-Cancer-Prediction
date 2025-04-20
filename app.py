from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import datetime
app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

column_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

@app.route('/')
def index():
    return render_template('index.html', column_names=column_names, datetime=datetime)

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']

    # Retrieve the feature values from the form
    features = [float(request.form[f'feature{i}']) for i in range(len(column_names))]

    # Convert into NumPy array for prediction
    input_data = np.array(features).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(input_data)
    result = "Malignant (Cancerous)" if prediction[0] == 1 else "Benign (Non-Cancerous)"

    return render_template('result.html', 
                         name=name, 
                         age=age, 
                         gender=gender,
                         features=features, 
                         column_names=column_names, 
                         result=result,
                         datetime=datetime)

if __name__ == '__main__':
    app.run(debug=True)
