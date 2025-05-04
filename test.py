import joblib
import re
from nltk.stem.porter import PorterStemmer
from xgboost import XGBClassifier
import numpy as np

tfid = joblib.load('saved_models/tfidf_vectorizer.pkl')

model = joblib.load('saved_models/RF_model.pkl')
# model = XGBClassifier()
# model.load_model('saved_models/xgb_model.json')
stemmer = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return " ".join([stemmer.stem(word) for word in text.split()])

def predict_message(text):
    processed = transform_text(text)
    vectorized = tfid.transform([processed]).toarray()
    pred = model.predict(vectorized)
    return "Spam" if pred[0] == 1 else "Not Spam"

if __name__ == "__main__":
    sample = input("Enter message: ")
    result = predict_message(sample)
    print("Result:", result)
