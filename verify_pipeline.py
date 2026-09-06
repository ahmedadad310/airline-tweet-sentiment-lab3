import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('stopwords', quiet=True)
df = pd.read_csv('Tweets.csv')
assert {'text', 'airline_sentiment'}.issubset(df.columns)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    return text.strip()

df['clean_text'] = df['text'].apply(clean_text)
stop_words = set(stopwords.words('english')) - {'not', 'no'}
df['clean_text'] = df['clean_text'].apply(lambda text: ' '.join(w for w in text.split() if w not in stop_words))
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['clean_text'])
y = df['airline_sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print('rows=', len(df))
print('classes=', sorted(y.unique().tolist()))
print('X_shape=', X.shape)
print('train_shape=', X_train.shape, 'test_shape=', X_test.shape)
print('accuracy=', round(accuracy_score(y_test, predictions), 4))
print('sample_prediction=', model.predict(vectorizer.transform(['great flight and amazing crew']))[0])
