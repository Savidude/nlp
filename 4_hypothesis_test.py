import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


reviews = pd.read_csv("data/derived/restaurant_reviews_textblob.csv", sep="\t")

# 1. cleaning the dataset (words like a, an, and , the which makes no sense in reviews)
nltk.download('stopwords')
corpus = []
for i in range(1000):
    review = re.sub('[^a-zA-Z]' , ' ', reviews['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    stop = stopwords.words('english')
    stop.remove('not')
    review = [ps.stem(word) for word in review if not word in set(stop)]
    review = " ".join(review)
    corpus.append(review)
print(corpus[0:10])

