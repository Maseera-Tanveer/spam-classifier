import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['spamORham', 'Message']]
data.columns = ['label', 'message']

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

model = MultinomialNB()
model.fit(X, y)

# UI
st.title("Spam Message Classifier")

user_input = st.text_area("Enter your message:")

if st.button("Check"):
    input_vector = vectorizer.transform([user_input])
    result = model.predict(input_vector)

    if result[0] == 1:
        st.error("Spam Message 🚨")
    else:
        st.success("Not Spam ✅")