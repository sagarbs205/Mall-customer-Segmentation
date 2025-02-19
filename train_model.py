import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\Admin\PycharmProjects\PythonProject3\Mall_Customer.csv")
df.head()

# statistical info
df.describe()
# datatype info
df.info()

import seaborn as sns
sns.countplot(x=df['Gender'])  # ✅ Works, but better to use 'data' argument
sns.distplot(df['Age'])

sns.distplot(df['Annual Income (k$)'])

sns.distplot(df['Spending Score (1-100)'])

corr = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')

df.head()

# cluster on 2 features
df1 = df[['Annual Income (k$)', 'Spending Score (1-100)']]
df1.head()

#import seaborn as sns

sns.scatterplot(x=df1['Annual Income (k$)'], y=df1['Spending Score (1-100)'])

from sklearn.cluster import KMeans
errors = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df1)
    errors.append(kmeans.inertia_)

# plot the results for elbow method
plt.figure(figsize=(13,6))
plt.plot(range(1,11), errors)
plt.plot(range(1,11), errors, linewidth=3, color='red', marker='8')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS')
plt.xticks(np.arange(1,11,1))
plt.show()

km = KMeans(n_clusters=5)
km.fit(df1)
y = km.predict(df1)
df1['Label'] = y
df1.head()

sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df1, hue='Label', s=50, palette=['red', 'green', 'brown', 'blue', 'orange'])

# cluster on 3 features
df2 = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]
df2.head()

errors = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df2)
    errors.append(kmeans.inertia_)

    # plot the results for elbow method
    plt.figure(figsize=(13, 6))
    plt.plot(range(1, 11), errors)
    plt.plot(range(1, 11), errors, linewidth=3, color='red', marker='8')
    plt.xlabel('No. of clusters')
    plt.ylabel('WCSS')
    plt.xticks(np.arange(1, 11, 1))
    plt.show()

    km = KMeans(n_clusters=5)
    km.fit(df2)
    y = km.predict(df2)
    df2['Label'] = y
    df2.head()

    # 3d scatter plot
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df2['Age'][df2['Label'] == 0], df2['Annual Income (k$)'][df2['Label'] == 0],
               df2['Spending Score (1-100)'][df2['Label'] == 0], c='red', s=50)
    ax.scatter(df2['Age'][df2['Label'] == 1], df2['Annual Income (k$)'][df2['Label'] == 1],
               df2['Spending Score (1-100)'][df2['Label'] == 1], c='green', s=50)
    ax.scatter(df2['Age'][df2['Label'] == 2], df2['Annual Income (k$)'][df2['Label'] == 2],
               df2['Spending Score (1-100)'][df2['Label'] == 2], c='blue', s=50)
    ax.scatter(df2['Age'][df2['Label'] == 3], df2['Annual Income (k$)'][df2['Label'] == 3],
               df2['Spending Score (1-100)'][df2['Label'] == 3], c='brown', s=50)
    ax.scatter(df2['Age'][df2['Label'] == 4], df2['Annual Income (k$)'][df2['Label'] == 4],
               df2['Spending Score (1-100)'][df2['Label'] == 4], c='orange', s=50)
    ax.view_init(30, 190)
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income')
    ax.set_zlabel('Spending Score')
    palette = ['red', 'green', 'brown', 'blue', 'orange']
    plt.show()

# ✅ Save Model using Pickle
    model_filename = "kmeans_model.pkl"
    with open(model_filename, "wb") as model_file:
        pickle.dump(kmeans, model_file)

print(f"\n✅ Model saved successfully as '{model_filename}'")