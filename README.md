# Mall Customer Segmentation using Machine Learning & Flask

## Overview
This project performs customer segmentation using machine learning techniques and integrates the results into a **Flask-based web application**. The goal is to categorize mall customers into different groups based on their spending patterns and annual income, helping businesses tailor their marketing strategies accordingly.

## Dataset
The dataset contains information on mall customers, including:
- Customer ID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## Objective
To segment customers into clusters based on their income and spending behavior using **K-Means Clustering** and **Hierarchical Clustering**, and present the results via a Flask web interface.

## Technologies Used
- Python
- Flask (Web Framework)
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- HTML/CSS (for Flask UI)

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mall-customer-segmentation.git
   cd mall-customer-segmentation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Open your browser and visit:
   ```
   http://127.0.0.1:5000/
   ```

## Implementation Steps
1. **Data Preprocessing**
   - Load the dataset
   - Handle missing values (if any)
   - Standardize features
2. **Exploratory Data Analysis (EDA)**
   - Visualize income and spending patterns
   - Detect potential customer clusters
3. **Applying Clustering Algorithms**
   - K-Means Clustering
   - Hierarchical Clustering
4. **Visualization & Insights**
   - Scatter plots of clusters
   - Dendrogram for hierarchical clustering
5. **Web Application Development with Flask**
   - Create a web interface for users to upload and analyze their data
   - Display visualizations of clustered customer segments
   - Allow users to input new customer data and predict their cluster
6. **Evaluation**
   - Use the **Elbow Method** to determine the optimal number of clusters
   - Interpret cluster characteristics

## Results & Findings
- Customers were successfully segmented into **5 distinct groups** based on their income and spending behavior.
- Visualizing clusters helped in identifying **high-spending** vs. **low-spending** groups.
- Businesses can target specific clusters with personalized marketing strategies.
- The Flask web application allows users to easily analyze their customer data without coding knowledge.

## Usage
- **Run the Flask app** and interact with the web interface to analyze customer segmentation.
- Modify the dataset or clustering parameters to customize the segmentation as needed.
- Deploy the Flask application on **Heroku, AWS, or PythonAnywhere** for online access.

## Contributing
Feel free to fork the repository and submit a pull request if you'd like to contribute improvements.

## License
This project is licensed under the MIT License.

---
**Author:** SAGAR B S 
**GitHub:** [sagarbs205](https://github.com/sagarbs205)

