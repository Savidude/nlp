import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud

from collections import Counter

stop_words = set(stopwords.words('english'))
punctuation = [".", ",", "!","n\'t","\'s"]

incorrectly_annotated_reviews = []
review_tokens = []
review_words = ""

reviews = pd.read_csv("data/derived/restaurant_reviews_textblob.csv", sep="\t")

for i, review in reviews.iterrows():
    liked = review['Liked']
    category = review['Category']

    if liked != category:
        incorrectly_annotated_reviews.append(review['Review'].lower())

for review in incorrectly_annotated_reviews:
    tokens = word_tokenize(review)
    review_tokens.extend(tokens)
    review_words += " ".join(tokens) + " "

'''
-------------------------------------- Generate Word Cloud --------------------------------------
'''
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stop_words,
                min_font_size = 10).generate(review_words)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()

'''
-------------------------------------- Generate Bar Graph --------------------------------------
'''
filtered_review_tokens = []
for token in review_tokens:
    if token not in stop_words and token not in punctuation:
        filtered_review_tokens.append(token)

word_counter = Counter(filtered_review_tokens)
most_occur = word_counter.most_common(10)

words = []
word_counts = []
for instance in most_occur:
    words.append(instance[0])
    word_counts.append(instance[1])
fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(words, word_counts, width=0.4)

plt.xlabel("Most common words")
plt.ylabel("Number of occurrences")
plt.title("The number of occurrences of the most common words")
plt.show()
