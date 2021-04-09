import numpy as np
import pandas as pd
import math
import random as rd
from random import seed
from random import gauss
import csv

class kmeans:
    def __init__(self, df, k):
        seed(1)
        self.data = pd.DataFrame.to_numpy(df)
        self.k = arr = np.empty(k , dtype=int)
        for i in range(0, len(self.k)):
            self.k[i] = i

    def kmean(self):
        x = True
        # random clustering as imposed by the professor
        clusters = np.zeros((len(self.k), len(self.data[0])))
        for i in range(0, len(clusters)):
            for x in range(0, len(clusters[i])):
                clusters[i][x] = gauss(0, 2)

        clusters = np.array(clusters)
        labels = np.zeros((self.data).T[0].shape)
        labels = np.array(labels)
        #loop until x becomes False, it'll become false when cluster means don't change
        #just and all the cluster assignements
        while (x):
            #iterate through all points and calculate the cluster assignement
            for i in range(0, len(self.data)):
                self.assign_cluster(self.data[i], clusters, labels, i)

            centroid = np.zeros(clusters.shape)



            #recompute centroid[i]
            for i in range(0, len(clusters)):
                denominator = 0
                #iterate through all the points
                for x in range(0, len(self.data)):
                    if(labels[x] == i):
                        centroid[i] = np.add(centroid[i], self.data[x])
                        denominator += 1
                if (denominator != 0):
                    centroid[i] = centroid[i]/denominator
                else:
                    centroid[i] = centroid[i]

            #self.compute_centroid(clusters, labels)


            if ((centroid == clusters).all()):
                x = False
                return labels
            else:
                x = True
                clusters = centroid

    #calculates the distance between a point and a centroid
    def euclidian_distance(self, point1, point2):
        distance = 0
        if (point1.shape == point2.shape):
            for i in range(0, len(point1)):
                distance += ((point1[i] - point2[i]))**2

        return math.sqrt(distance)

    #gets minimum euclidian distance between a point and each cluster,
    #and labels the correct index in the array of clusters
    def assign_cluster(self, point, clusters, labels, index):
        closest = float('inf')
        decision = 0
        for i in range(0, len(clusters)):
            x = self.euclidian_distance(point, clusters[i])
            if (x < closest):
                closest = x
                decision = i
        labels[index] = decision


    #computes the new cluster values after each assignement
    def compute_centroid(self, clusters, labels):

        for i in range(0, len(clusters)):
            centroid = np.zeros(clusters.shape)
            denominator = 0
            # iterate through all the points
            for x in range(0, len(self.data)):
                if (labels[x] == i):
                    centroid[i] = np.add(centroid[i], self.data[x])
                    denominator += 1
            if (denominator != 0):
                centroid[i] = centroid[i] / denominator
            else:
                centroid[i] = centroid[i]

    def load_data(self, path):
        x = np.genfromtxt(path, delimiter='\t')
        return x

#x = kmeans('Data.tsv', 3)
#clusters = x.kmean()
#with open('kmeans_output.tsv', 'w', newline='') as f_output:
#    tsv_output = csv.writer(f_output, delimiter='\t')
#    tsv_output.writerow(clusters)
