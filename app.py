from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained KMeans model
model_path = r"C:\Users\Admin\PycharmProjects\PythonProject3\model\kmeans_model.pkl"
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the form
        age = float(request.form['age'])
        income = float(request.form['income'])
        spending_score = float(request.form['spending_score'])

        # Convert inputs to a numpy array
        user_data = np.array([[income, spending_score, age]])

        # Predict the cluster
        cluster = model.predict(user_data)[0]

        return render_template('index.html', prediction=f'Customer belongs to Cluster {cluster}')

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
