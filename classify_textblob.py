import pandas as pd
import csv

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


# import nltk
# nltk.download('movie_reviews')

def categorize_sentiment(sentiment):
    if sentiment == "pos":
        return 1
    else:
        return 0


reviews = pd.read_csv("data/Restaurant_Reviews.tsv", sep="\t")

with open('data/derived/restaurant_reviews_textblob.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter="\t")
    writer.writerow(["Review", "Liked", "Category"])

    for i, review in reviews.iterrows():
        text = review['Review']
        blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
        sentiment = blob.sentiment
        category = categorize_sentiment(sentiment.classification)

        writer.writerow([text, review['Liked'], category])
        print(text, " -> ", sentiment.classification)
