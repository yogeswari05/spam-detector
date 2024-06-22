import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Load the trained model
with open('spam_detect_model.pkl', 'rb') as file:
   spam_detect_model = pickle.load(file)

# Load the CountVectorizer (assuming it was fitted during training)
with open('count_vectorizer.pkl', 'rb') as file:
   vectorizer = pickle.load(file)

# Define the prediction function
def predict_spam(text):
   transformed_text = vectorizer.transform([text])
   prediction = spam_detect_model.predict(transformed_text)
   return prediction

# Streamlit app interface
st.title("Spam Detection App")

# Text input
user_input = st.text_area("Enter the message:")

if st.button("Predict"):
   prediction = predict_spam(user_input)
   if prediction == 1:
      st.write("This message is classified as **Spam**.")
   else:
      st.write("This message is classified as **Not Spam**.")
