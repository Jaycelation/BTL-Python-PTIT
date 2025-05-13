# SMS Phishing Detection

## Overview

This project is a Python-based implementation for detecting phishing SMS messages as part of a coursework assignment (BTL) at PTIT (Posts and Telecommunications Institute of Technology). The goal is to classify SMS messages into two categories: **ham** (legitimate) and **spam** (phishing) using various machine learning models. The project leverages Natural Language Processing (NLP) techniques and machine learning algorithms to achieve high accuracy in identifying phishing messages.

## Key Features

- **Dataset**: Utilizes the [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) dataset with 5,572 SMS messages (86.6% ham, 13.4% spam).
- **Preprocessing**: Cleans and transforms text data using tokenization, stop words removal, and stemming.
- **Feature Extraction**: Employs `TfidfVectorizer` for converting text into numerical features.
- **Models**: Evaluates multiple machine learning models, including Naive Bayes, Logistic Regression, SVM, Random Forest, and more.
- **Performance**: Achieves high accuracy (up to 97.6% with Random Forest) and precision on the test set.

## Requirements

To run this project, you need the following Python libraries:

- numpy
- pandas
- scikit-learn
- matplotlib
- nltk
- wordcloud
- jinja2
- seaborn
- xgboost

You can install all dependencies by running:

```bash
pip install -r requirements.txt
```

Additionally, download NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Project Structure

```
BTL-Python-PTIT/
│
├── data/
│   └── spam.csv              # Dataset file
├── saved_models/             # Directory for saved models and TF-IDF vectorizer
│   ├── RF_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── ...
├── model.ipynb               # Main Jupyter Notebook with implementation
├── requirements.txt          # List of dependencies
└── README.md                 # Project documentation
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Jaycelation/BTL-Python-PTIT.git
   cd BTL-Python-PTIT
   ```

2. **Set up a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**:

   - The dataset (`spam.csv`) is included in the `data/` folder. Alternatively, download it from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

5. **Run the Jupyter Notebook**:

   ```bash
   jupyter notebook model.ipynb
   ```

## Usage

1. **Data Preprocessing**:

   - The dataset is loaded from `spam.csv`.
   - Unnecessary columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`) are dropped.
   - Columns are renamed (`v1` to `target`, `v2` to `text`).
   - Labels are encoded (`ham` → 0, `spam` → 1) using `LabelEncoder`.

2. **Feature Extraction**:

   - Text is preprocessed (lowercase, remove punctuation, tokenization, remove stop words).
   - `TfidfVectorizer` with `max_features=3000` is used to convert text into TF-IDF features.

3. **Model Training**:

   - The dataset is split into 80% training and 20% testing sets.
   - Multiple models are trained and evaluated:
     - Support Vector Classifier (SVC)
     - K-Nearest Neighbors (KNN)
     - Multinomial Naive Bayes (NB)
     - Decision Tree (DT)
     - Logistic Regression (LR)
     - Random Forest (RF)
     - AdaBoost
     - Bagging Classifier
     - Extra Trees Classifier (ETC)
     - Gradient Boosting (GBDT)
     - XGBoost

4. **Evaluation**:

   - Models are evaluated using accuracy, precision, F1-score, and confusion matrix.
   - Results are printed to the console and saved to `results.txt`.

5. **Prediction**:

   - Load a saved model (e.g., Random Forest) and the fitted `TfidfVectorizer`:

     ```python
     import joblib
     rf_model = joblib.load('saved_models/RF_model.pkl')
     tfid = joblib.load('saved_models/tfidf_vectorizer.pkl')
     ```

   - Predict on new text:

     ```python
     text = ["You have won a lottery worth $1000! Claim now."]
     vectorized = tfid.transform(text).toarray()
     prediction = rf_model.predict(vectorized)
     print("Prediction:", prediction[0])  # 0: ham, 1: spam
     ```

## Results

The project evaluates 11 machine learning models on the test set. Below are the key performance metrics:

| Model             | Accuracy | Precision (Weighted) | F1-Score (Weighted) |
|-------------------|----------|---------------------|---------------------|
| SVC               | 97.52%   | 97.55%              | 97.39%              |
| KNN               | 91.85%   | 92.53%              | 89.57%              |
| Naive Bayes       | 97.32%   | 97.36%              | 97.16%              |
| Decision Tree     | 93.91%   | 93.47%              | 93.44%              |
| Logistic Regression | 96.08%   | 95.97%              | 95.83%              |
| Random Forest     | **97.63%** | **97.63%**          | **97.52%**          |
| AdaBoost          | 93.40%   | 93.21%              | 92.35%              |
| Bagging           | 96.90%   | 96.84%              | 96.86%              |
| Extra Trees       | 97.52%   | 97.48%              | 97.44%              |
| Gradient Boosting | 95.87%   | 95.79%              | 95.56%              |
| XGBoost           | 97.32%   | 97.26%              | 97.22%              |

**Key Observations**:

- **Random Forest** achieves the highest accuracy (97.63%) and F1-score (97.52%), making it the best-performing model.
- **Naive Bayes** and **SVC** also perform exceptionally well, with accuracies above 97%.
- **KNN** and **Decision Tree** have lower performance, likely due to the high-dimensional nature of TF-IDF features.

## Future Improvements

- **Advanced Preprocessing**: Incorporate lemmatization or more sophisticated text cleaning techniques.
- **Deep Learning**: Experiment with deep learning models like LSTM or BERT for better semantic understanding.
- **Feature Engineering**: Explore n-grams or word embeddings (e.g., Word2Vec, GloVe) to capture more context.
- **Real-Time Deployment**: Integrate the model into a web or mobile application for real-time SMS filtering.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements. Ensure that your code follows the project's coding style and includes appropriate tests.

## Acknowledgments

- [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) for providing the data.
- [scikit-learn](https://scikit-learn.org/) for machine learning tools.
- [NLTK](https://www.nltk.org/) for NLP preprocessing.
