import Clean_Article as ca
import numpy as np
import pandas as pd
import GMDC as gmdc
import random as rd
import KMeans as km
import pythonScraper as ps


dataset = np.genfromtxt("Articles.txt", delimiter='\n')
dataset = open("Articles.txt")
contents = dataset.read()
articles = contents.splitlines()

x = ca.ScrapedArticles(articles)
x.clean_database()
x.find_unique_words()
df = x.computeTF_IDF()
df = pd.DataFrame(x.computeTF_IDF())
#x = x.dimension_reduction(df)

#y = km.kmeans(df, 2)
#clusters = y.kmean()
z = gmdc.GMM(df, 2)
clusters = z.GMM()

print(clusters)
