import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Select columns
data = data[['spamORham', 'Message']]
data.columns = ['label', 'message']

# Convert labels to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])

y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test model
accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)

# Test with your own message
sample = ["you won a gift share your upi pin we will transfer the amount"]

sample_vector = vectorizer.transform(sample)

prediction = model.predict(sample_vector)

if prediction[0] == 1:
    print("Spam")
else:
    print("Not Spam")