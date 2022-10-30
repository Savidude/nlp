import pandas as pd
import csv
from sentistrength import PySentiStr

senti = PySentiStr()
senti.setSentiStrengthPath('C:/SentiStrength.jar')
senti.setSentiStrengthLanguageFolderPath('C:/SentStrength_Data/')

reviews = pd.read_csv("data/Restaurant_Reviews.tsv", sep="\t")

with open('data/derived/restaurant_reviews_sentistrength.csv', 'w', newline='', encoding='UTF-8') as file:
    writer = csv.writer(file, delimiter="\t")
    writer.writerow(["Review", "Liked", "Category"])

    for i, review in reviews.iterrows():
        text = review['Review']
        category = senti.getSentiment(text, score='binary')
        if str(category) == '[-1]':
            category = '0'
        else:
            category = '1'

        writer.writerow([text, review['Liked'], category])
        print(text, " -> ", str(category))
