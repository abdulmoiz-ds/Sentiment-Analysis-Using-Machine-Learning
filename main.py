import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
movie_reviews = pd.read_csv('movie_reviews.csv')
stop_words = set(stopwords.words('english'))

vectorizer = CountVectorizer(stop_words=stop_words)

X = vectorizer.fit_transform(movie_reviews['review'])
y = movie_reviews['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print("Accuracy:", accuracy)
