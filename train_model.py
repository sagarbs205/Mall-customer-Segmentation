import os

os.environ['OMP_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("C:/Users/Admin/PycharmProjects/PythonProject3/Mall_Customer.csv")

# Select relevant features
df_model = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]

# Train KMeans model
kmeans = KMeans(n_clusters=5,n_init='auto')
kmeans.fit(df_model)

# Save the trained model as a pickle file
with open('model/kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)

print("Model saved successfully as 'kmeans_model.pkl'")
