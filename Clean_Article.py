import numpy as np
import nltk
import re
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class ScrapedArticles():
    def __init__(self, articles_list):
        self.database = articles_list
        self.bagsOfWords = []
        self.wordFreq = [{}]

    def clean_database(self):
        self.database.remove('')
        #remove non-letters
        for i in range(0, len(self.database)):
            exclusion_list = ['[^a-zA-Z]', 'rt', 'http', 'co', 'RT']
            exclusions = '|'.join(exclusion_list)
            self.database[i] = re.sub(exclusions, ' ', self.database[i])
            self.database[i] = self.database[i].lower()

        #create a bag of words
        for i in range(0, len(self.database)):
            self.bagsOfWords.append(
            word_tokenize(self.database[i])
            )
            bag = [x for x in self.bagsOfWords[i] if ((not (x in stopwords.words('english'))) and x != "") ]
            self.bagsOfWords[i] = bag

        #lemmatize to keep 'root' of words
        lemmatizer = WordNetLemmatizer()
        for s in range(0, len(self.bagsOfWords)):
            for i in range(0, len(self.bagsOfWords[s])):
                word = self.bagsOfWords[s][i]
                self.bagsOfWords[s][i] = lemmatizer.lemmatize(word, self.get_wordnet_pos(word))


    def find_unique_words(self):

        #find all the words used in the articles and puts it in list
        unique_words = []
        for i in self.bagsOfWords:
            unique_words = set(unique_words).union(set(i))


        #count all the times a word appears in a doccument
        word_count = []
        for s in range (0, len(self.bagsOfWords)):
            word_count.append(dict.fromkeys(unique_words, 0))
            for i in self.bagsOfWords[s]:
                word_count[s][i] += 1
        self.wordFreq = word_count

        #apply sqrt to reduce influence of high count
        for s in range(0, len(self.wordFreq)):
            for word in self.wordFreq[s]:
                count = self.wordFreq[s][word]
                self.wordFreq[s][word] = np.sqrt(count)

    def computeTF(self) -> [{}]:
        tfDict = []
        for i in range (0, len(self.wordFreq)):
            tfDict.append({})
            for word, count in self.wordFreq[i].items():
                tfDict[i][word] = count/float(len(self.bagsOfWords[i]))
        return tfDict

    def computeIDF(self) -> [{}]:
        idf = {}
        x = 0
        for word in self.wordFreq[0]:
            for i in range(0, len(self.wordFreq)):
                if (self.wordFreq[i].get(word, 0) != 0):
                    idf[word] = idf.get(word, 0) + 1
            idf[word] = len(self.bagsOfWords)/idf.get(word, 0)
            x += 1
        return idf

    def computeTF_IDF(self):
        tf = self.computeTF()
        idf = self.computeIDF()

        tf_idf = []
        for i in range(0, len(self.wordFreq)):
            tf_idf.append({})
            for word in self.wordFreq[0]:
                tf_idf[i][word] = tf[i].get(word, 0)*idf.get(word, 0)
        return tf_idf

    def dimension_reduction(self, df):
        df = StandardScaler().fit_transform(df)
        pca = PCA(n_components=5)

        principalComponents = pca.fit_transform(df)
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5'])
        return principalDf

    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)







