import pandas as pd
import numpy as np
from empath import Empath
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
lexicon = Empath()
import requests
import  json

backend_url="http://54.148.189.209:8000"

price_cat = requests.post(backend_url + "/create_category", json={"terms":["price"],"size":100,"model":'fiction'})
price_cat = json.loads(price_cat.text)
#to save the result vector in price_cat created from lexicon.create_category("price",["price"])

location_cat = requests.post(backend_url + "/create_category", json={"terms":["location"],"size":100,"model":'fiction'})
location_cat = json.loads(location_cat.text)
#lexicon.create_category("location",["location"])

quality_cat = requests.post(backend_url + "/create_category", json={"terms":["quality"],"size":100,"model":'fiction'})
quality_cat = json.loads(quality_cat.text)
#lexicon.create_category("quality",["quality"])

quantity_cat = requests.post(backend_url + "/create_category", json={"terms":["quantity"],"size":100,"model":'fiction'})
quantity_cat = json.loads(quantity_cat.text)
#lexicon.create_category("quantity",["quantity"])

#reading data
reviews = pd.read_csv("data/derived/restaurant_reviews_textblob.csv", sep="\t")

#function for calculating cosine similarity
def get_similarity(cat, data):
    # sw contains the list of stopwords
    sw = stopwords.words('english')
    l1 = []
    l2 = []

    # remove stop words from the string
    X_set = {w for w in cat if not w in sw}
    Y_set = {w for w in data if not w in sw}

    # form a set containing keywords of both strings
    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set:
            l1.append(1)  # create a vector
        else:
            l1.append(0)
        if w in Y_set:
            l2.append(1)
        else:
            l2.append(0)
    c = 0

    # cosine similarity
    for i in range(len(rvector)):
        c += l1[i] * l2[i]
    cosine = c / float((sum(l1) * sum(l2)) ** 0.5)

    return cosine


for i in range(1000):
    data = word_tokenize(reviews['Review'][i])
    simliarities = []
    price_cosine = get_similarity(price_cat, data)
    location_cosine = get_similarity(location_cat, data)
    quantity_cosine = get_similarity(quantity_cat, data)
    quality_cosine = get_similarity(quality_cat, data)
    print()


    simliarities.append(price_cosine)
    simliarities.append(location_cosine)
    simliarities.append(quantity_cosine)
    simliarities.append(quality_cosine)

    categories = ['Price', 'location', 'quantity', 'quality']
    maxi = categories[np.argmax(simliarities)] #getting max similarity as the most relavent catagory for a particular review for the data

    #print results
    print("review index:", i)
    print("cosine similarity between review and catagories:","[", "price:", ",", simliarities[0],",","location:", simliarities[1],",","quantity:", simliarities[2],",","quality:", simliarities[3],"]" )

    if (np.argmax(simliarities) == 0.0) :
        print("This review is not from four catagories")
    else:
        print("appropriate category is", maxi)

    print("Results from Lexicon analysis:", lexicon.analyze(data, categories=["price","location","quantity","quality"], normalize=True))









