import numpy as np
import pandas as pd

reviews = pd.read_csv("data/derived/restaurant_reviews_textblob.csv", sep="\t")

x, y = [], []
for i, review in reviews.iterrows():
    x.append(int(review['Liked']))
    y.append(int(review['Category']))

# returns correlation matrix,
# which is a two-dimensional
# array with the correlation coefficients.
r = np.corrcoef(x, y)

print(f'Pearson correlation coefficient: {r[0, 1]}')
# Current output: +0.27443338505026915, which means that
# there is a small correlation between sentiment polarity
# and the annotation.
