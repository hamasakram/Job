# Job Recommendation System

## Project Overview
This project is a **Job Recommendation System** that leverages machine learning to suggest job titles based on user input. It processes textual data, such as skills, experience, and preferences, and uses a trained neural network model to recommend job titles.

## Features
- **Preprocessing:** Cleans and vectorizes text data from job descriptions.
- **Model Training:** Implements a neural network using TensorFlow for classification.
- **Real-Time Predictions:** Provides job recommendations through a Streamlit web interface.

## Requirements
- Python 3.x
- TensorFlow
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
# Usage
Launch the Streamlit app:
streamlit run app.py

Open the app in your browser and enter your skills, experience, and preferences to get a job recommendation.
Dataset
The system uses final_data.csv, which contains job descriptions, skills, qualifications, and other related information. The dataset is preprocessed to create a combined text feature for machine learning.

## Key Components
# Data Preprocessing:

Removes HTML tags and punctuation.
Combines multiple text features into a single column.
Vectorizes text using TF-IDF.
# Model Training:

A feedforward neural network with:
Input layer for TF-IDF features.
Two hidden layers with dropout for regularization.
Output layer for multi-class classification.
Uses Adam optimizer and sparse categorical cross-entropy loss.
# Prediction:
The model predicts the most suitable job title based on user input.
# Example Inputs
Skills: "Python, Data Analysis, Machine Learning"
Experience: "3 years in data science"
Preferences: "Remote work"

# Outputs
The system outputs the most relevant job title based on the input data.
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hamasakram/Job.git
