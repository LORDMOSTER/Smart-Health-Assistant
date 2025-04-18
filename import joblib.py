import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Sample dataset — replace with your actual symptoms and diseases
data = {
    'symptoms': [
        'fever, cough, sore throat',
        'headache, nausea',
        'skin rash, itching',
        'fever, fatigue, joint pain'
    ],
    'disease': ['Flu', 'Migraine', 'Allergy', 'Dengue']
}

df = pd.DataFrame(data)

# Vectorize symptoms
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['symptoms'])
y = df['disease']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, 'disease_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved!")
