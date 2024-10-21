# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib  # To save or load models

# Define a function to load and preprocess data
@st.cache(allow_output_mutation=True)
def load_data():
    dt = pd.read_csv('final_data.csv')
    
    # Cleaning text data
    def clean_text(text):
        if isinstance(text, str):
            text = re.sub('<.*?>', '', text)  # Remove HTML tags
            text = re.sub('[^\w\s]', '', text)  # Remove punctuation
            text = re.sub('\s+', ' ', text)  # Replace multiple spaces with a single space
            text = text.strip()  # Remove leading and trailing whitespace
        return text

    text_columns = ['Job Description', 'Qualifications', 'skills', 'Responsibilities']
    for column in text_columns:
        dt[column] = dt[column].apply(clean_text)

    dt.dropna(inplace=True)

    # Combining text data into a single column
    dt['combined_text'] = dt['skills'] + ' ' + dt['Experience'] + ' ' + dt['Preference'] + ' ' + dt['Qualifications']
    
    # Vectorizing text data
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
    X_features = tfidf_vectorizer.fit_transform(dt['combined_text'])
    
    # Encoding job titles
    label_encoder = LabelEncoder()
    y_labels = label_encoder.fit_transform(dt['Job Title'])
    
    return X_features, y_labels, tfidf_vectorizer, label_encoder, dt

X_features, y_labels, tfidf_vectorizer, label_encoder, dt = load_data()

# Load or train model
def load_or_train_model(X_features, y_labels):
    try:
        # Try to load a pre-trained model
        model = tf.keras.models.load_model('job_recommendation_model.h5')
    except:
        # If not available, train the model
        X_train, X_val, y_train, y_val = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)
        model = Sequential([
            Dense(256, activation='relu', input_dim=X_features.shape[1]),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(len(label_encoder.classes_), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
        model.save('job_recommendation_model.h5')  # Save the model for future use
    return model

model = load_or_train_model(X_features, y_labels)

# Function to recommend job title based on inputs
def recommend_job_title(skills, experience, preferences):
    combined_input = f"{skills} {experience} {preferences}"
    input_vector = tfidf_vectorizer.transform([combined_input])
    prediction = model.predict(input_vector)
    predicted_job_title = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_job_title

# Streamlit interface
st.title('Job Recommendation System')
user_skills = st.text_input("Enter your skills")
user_experience = st.text_input("Enter your experience")
user_preferences = st.text_input("Enter your preferences")

if st.button('Recommend a Job'):
    recommended_job = recommend_job_title(user_skills, user_experience, user_preferences)
    st.success(f"Recommended Job Title: {recommended_job}")
