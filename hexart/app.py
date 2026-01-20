import pickle
import numpy as np
import os
from flask import Flask, render_template, request, jsonify

# Load the trained model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR,"breast_cancer_model.pkl")

with open(model_path,"rb") as file:
    model = pickle.load(file)

# Feature names for input
feature_names = [
    'Clump Thickness', 'Uniform Cell Size', 'Uniform Cell Shape',
    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
    'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'
]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from the form
        form_data = request.form.to_dict()
        form_data = {key: float(value) for key, value in form_data.items()}

        # Prepare the data for prediction
        data = np.array(list(form_data.values())).reshape(1, -1)

        # Predict using the trained model
        prediction = model.predict(data)

        # Return prediction as 'You have cancer' or 'You don't have cancer'
        result = 'You have cancer' if prediction[0] == 0 else 'You don\'t have cancer'
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
