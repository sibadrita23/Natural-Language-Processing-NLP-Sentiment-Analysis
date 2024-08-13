import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from src.data.preprocess_text import clean_text

def train_sentiment_model(csv_file):
    # Load and preprocess data
    df = pd.read_csv(csv_file)
    df['Review'] = df['Review'].apply(clean_text)

    # Vectorize text
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(df['Review']).toarray()
    y = df['Sentiment']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train model
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = classifier.predict(X_test)
    report = classification_report(y_test, y_pred)

    # Save model and vectorizer
    import pickle
    with open('models/sentiment_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    with open('models/count_vectorizer.pkl', 'wb') as f:
        pickle.dump(cv, f)

    return report

if __name__ == '__main__':
    print(train_sentiment_model('data/processed/reviews.csv'))
