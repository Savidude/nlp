import numpy as np
from numpy.linalg import norm
import pandas as pd

reviews = pd.read_csv("data/derived/restaurant_reviews_textblob.csv", sep="\t")

x, y = [], []
x_pos, y_pos = [], []
x_neg, y_neg = [], []
for i, review in reviews.iterrows():
    x.append(int(review['Liked']))
    y.append(int(review['Category']))
    if review['Liked'] == 1:
        x_pos.append(review['Liked'])
        y_pos.append(review['Category'])
    else:
        x_neg.append(review['Liked'])
        y_neg.append(review['Category'])

# compute cosine similarity
cosine = np.dot(x, y) / (norm(x) * norm(y))
print("Cosine Similarity:", cosine)

cosine_pos = np.dot(x_pos, y_pos) / (norm(x_pos) * norm(y_pos))
print("Cosine Similarity of the positive class:", cosine_pos)

# As x_neg = [0, 0, 0, 0, 0......] will be all zeros Cosine similarity of negative
# classes will be 0.0

# cosine_neg = np.dot(x_neg, y_neg) / (norm(x_neg) * norm(y_neg))
# print("Cosine Similarity of the negative class:", cosine_neg)
