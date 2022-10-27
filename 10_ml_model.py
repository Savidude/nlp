import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

# def print_stopwords(review):
#     review_tokens = word_tokenize(review)
#     review_stopwords = list(stop_words.intersection(review_tokens))
#     print(review, " --> ", review_stopwords)

stop_words = set(stopwords.words('english'))

reviews = pd.read_csv("data/derived/restaurant_reviews_textblob.csv", sep="\t")

# incorrectly_annotated_reviews = []
# for i, review in reviews.iterrows():
#     liked = review['Liked']
#     category = review['Category']
#
#     if liked != category:
#         incorrectly_annotated_reviews.append(review['Review'])
#
# for review in incorrectly_annotated_reviews:
#     print_stopwords(review)

# stopwords to remove: not, only,
ignored_stopwords = ['not', 'down', 'before', 'over', 'more', 'off']
for word in ignored_stopwords:
    stop_words.remove(word)

corpus = []
for i, review in reviews.iterrows():
    corpus.append(review['Review'])

# vectorizer = TfidfVectorizer(max_features=1000, min_df=0.6, stop_words=stop_words)
vectorizer = TfidfVectorizer(max_features=1000, stop_words=stop_words)
X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

doc_list = []
for i in range(len(corpus)):
    doc_list.append(str(i))
docs_tfidf = pd.DataFrame(X.todense(), index=doc_list, columns=feature_names)

liked = reviews['Liked']

X_train, X_test, y_train, y_test = train_test_split(docs_tfidf, liked, test_size=0.3)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

target_names = ['Liked', 'Not Liked']
print(classification_report(y_test, y_pred, target_names=target_names))
