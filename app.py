from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained KMeans model
with open('model/kmeans_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        annual_income = float(request.form['income'])
        spending_score = float(request.form['score'])
        age = float(request.form['age'])

        # Prepare input for model
        features = np.array([[annual_income, spending_score, age]])

        # Predict cluster
        cluster = model.predict(features)[0]

        return render_template('index.html', prediction=f'Customer belongs to Cluster {cluster}')

    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)
