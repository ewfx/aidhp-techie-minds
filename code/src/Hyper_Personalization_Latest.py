#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install flask flask-socketio pandas numpy scikit-learn transformers joblib aif360


# In[3]:


pip install --upgrade Flask


# In[4]:


import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StandardDataset
import joblib

# Initialize Flask App
app = Flask(__name__)
socketio = SocketIO(app)

# Data Processing Class
class DataProcessor:
    def __init__(self, file1, file2, file3, file4):
        self.df1 = pd.read_csv(file1)
        self.df2 = pd.read_csv(file2)
        self.df3 = pd.read_csv(file3)
        self.df4 = pd.read_csv(file4)
        self.df = pd.concat([self.df1, self.df2, self.df3, self.df4], ignore_index=True)
        self.label_encoders = {}
        self.scaler = MinMaxScaler()

    def encode_features(self):
        categorical_columns = ['Industry', 'Financial Needs', 'Preferences', 'Gender', 'Location', 'Interest', 'Education', 'Occupation', 'Platform', 'Transaction Type', 'Category', 'Payment_mode']
        for col in categorical_columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le

    def normalize_features(self):
        numeric_cols = ['Age', 'Incomeper year(in dollar)', 'Revenue(in dollars)', 'Amount(in Dollar)', 'Sentiment_Score']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)  # Handle NaN values
        self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])

    def process(self):
        self.encode_features()
        self.normalize_features()
        return self.df

# Sentiment Analysis Class
class SentimentAnalyzer:
    def __init__(self):
        self.model = pipeline("sentiment-analysis")

    def analyze(self, text):
        return self.model(text)[0]['label'] if pd.notna(text) else 'neutral'

# Recommendation Model Class
class RecommendationSystem:
    def __init__(self, df):
        self.df = df
        self.model = NearestNeighbors(n_neighbors=5, metric='cosine')

    def train(self):
        feature_cols = ['Age', 'Incomeper year(in dollar)', 'Revenue(in dollars)', 'Amount(in Dollar)', 'Sentiment_Score']
        if all(col in self.df.columns for col in feature_cols):
            self.df[feature_cols] = self.df[feature_cols].fillna(self.df[feature_cols].median())  # Handle NaN values
            self.model.fit(self.df[feature_cols])
            joblib.dump(self.model, 'recommendation_model.pkl')

    def load_model(self):
        if os.path.exists('recommendation_model.pkl'):
            self.model = joblib.load('recommendation_model.pkl')

    def get_recommendations(self, user_id):
        if user_id not in self.df['Customer_Id'].values:
            return []
        user_data = self.df[self.df['Customer_Id'] == user_id].iloc[:, :-1]
        distances, indices = self.model.kneighbors(user_data)
        return self.df.iloc[indices[0]]['Product_Id'].tolist()



# Bias Detection Class
class BiasDetector:
    def __init__(self, df):
        self.df = df.copy()

        # Encode Gender: Male → 1, Female → 0
        if 'Gender' in self.df.columns:
            self.df['Gender'] = self.df['Gender'].map({'Male': 1, 'Female': 0})

        # Define privileged (Male) & unprivileged (Female) groups
        privileged_groups = [{'Gender': 1}]
        unprivileged_groups = [{'Gender': 0}]

        # Ensure no NaN values
        self.df.fillna({'Gender': 0}, inplace=True)

        # Create StandardDataset
        self.dataset = StandardDataset(
            self.df,
            label_name='Sentiment_Label',
            protected_attribute_names=['Gender'],
            favorable_classes=['positive'],  # Adjust this based on actual labels
            privileged_classes=[[1]],  # Male = 1
        )

        self.bias_model = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    def detect_bias(self):
        return self.bias_model.fit_transform(self.dataset)


# Load and process data
data_processor = DataProcessor('Customer_Profile_Org.csv', 'Customer_Profile_Individual.csv', 'Social_Media_Sentiment.csv', 'Transaction_History.csv')
df = data_processor.process()

# Apply sentiment analysis
sentiment_analyzer = SentimentAnalyzer()
df['Sentiment_Label'] = df['Content'].apply(lambda x: sentiment_analyzer.analyze(x))

# Train and load recommendation model
recommender = RecommendationSystem(df)
recommender.train()
recommender.load_model()

# Detect bias
bias_detector = BiasDetector(df)
adjusted_dataset = bias_detector.detect_bias()

@app.route('/')
def home():
    return render_template('Dashboard.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json['Customer_Id']
    recommendations = recommender.get_recommendations(user_id)
    socketio.emit('update_recommendations', {'recommendations': recommendations})
    return jsonify({'recommendations': recommendations})


if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)



# In[ ]:


#flask run --host=0.0.0.0 --port=5000

