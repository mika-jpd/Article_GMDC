import numpy as np
import pandas as pd
import math
import KMeans as km
import csv
import KMeans as km


class GMM:
    def __init__(self, df, k):
        self.df = df
        self.data = pd.DataFrame.to_numpy(df)

        #clustering numbers
        self.k = np.empty(k , dtype=int)
        for i in range(0, k):
            self.k[i] = i

    def GMM(self):

        #innitialization
        log_likes = np.zeros(len(self.k))

        mix_coef = np.zeros(len(self.k))
        mean = np.zeros((len(self.k),len(self.data[0])))
        labels = km.kmeans(self.df, len(self.k))
        labels = labels.kmean()
        lam = np.zeros(len(self.k))

        cov = []
        for i in range(0, len(self.k)):
            cov.append(np.zeros((len(self.data[0]), len(self.data[0]))))

        '''
                #init the mix coeficients
                prob_start = 1/len(self.k)
                mix_coef = np.array(mix_coef)


                #init the clusters & labels
                cluster_start = np.array_split(self.data, len(self.k))


                for i in range(0, len(self.k)):
                    mean[i] = self.init_mean(cluster_start[i])
                    mix_coef[i] = prob_start
        '''

        self.optimize_mean(self.data, mean, cov, mix_coef, labels)
        self.optimize_mix_coef(self.data, mix_coef, mean, cov, labels)
        self.optimize_cov(self.data, mean, cov, mix_coef, labels)
        self.optimize_lambda(self.data, mean, labels, lam)

        x = True

        # calculate log likelihood for comparison after the iteration
        log_l_prev = self.log_likelihood(self.data, mean, cov, lam,mix_coef, labels)
        #while loop until convergence
        while(x):
            #assign cluster to each point
            for i in range(0, len(self.data)):
                labels[i] = self.assign_cluster(self.data[i], mean, cov, mix_coef, lam)

            self.optimize_mean(self.data, mean, cov, mix_coef, labels)
            self.optimize_cov(self.data, mean, cov, mix_coef, labels)
            self.optimize_mix_coef(self.data, mix_coef, mean, cov, labels)
            self.optimize_lambda(self.data, mean, labels,lam)

            log_l_post = self.log_likelihood(self.data, mean, cov, lam,mix_coef, labels)
            if(math.fabs((log_l_post-log_l_prev))<(10**(-5))):
                x = False
                return labels
            else:
                log_l_prev = log_l_post

    #finds the mean of a cluster given an entire set of datapoints
    def optimize_mean(self, data, mean, cov, mix_coef, labels):
        #iterate through all the clusters to find each clusters mean
        for k in range(0, len(mean)):
            numerator = 0
            denominator = 0
            for i in range(0, len(data)):
                denominator += self.zik_indicator(i, k, labels)
            #iterate through all the points
            for i in range(0, len(data)):
                numerator += self.zik_indicator(i, k, labels)*data[i]
            mean[k] = numerator/denominator

    #finds the cov of a cluster
    def optimize_cov(self, data, mean, cov, mix_coef, labels):
        #iterate through all the clusters
        for k in range(0, len(cov)):
            w = 0
            diag_k = 0
            # iterate through all the points
            for i in range(0, len(data)):
                w += self.zik_indicator(i, k, labels)*np.subtract(self.data[i], mean[k])*np.subtract(self.data[i], mean[k])[np.newaxis,:].T

            w = np.diag(w)
            w_diag = np.ones(len(w))
            for x in range(0, len(w_diag)):
                if(w[x] == 0):
                    w_diag[x] = 0.001
                else:
                    w_diag[x] = w[x]

            diag_k = np.diag(w_diag)/np.linalg.det(np.diag(w_diag))**(1/len(self.data[0]))
            cov[k] = diag_k

    def optimize_lambda(self, data, mean, labels, lam):
        for k in range(0, len(mean)):
            w_k = 0
            n_k = self.calculate_n_k(data, labels, k)
            for i in range(0, len(data)):
                w_k += self.zik_indicator(i, k, labels) * np.subtract(self.data[i], mean[k]) * np.subtract(self.data[i], mean[k])[np.newaxis, :].T
            w_k = np.diag(w_k)
            w_diag = np.ones(len(w_k))
            for x in range(0, len(w_diag)):
                if (w_k[x] == 0):
                    w_diag[x] = 0.001
                else:
                    w_diag[x] = w_k[x]
            lam[k] = (1/n_k)*np.linalg.det(np.diag(w_diag))**(1/len(self.data[0]))

    #finds the mix_coef of each custer, need to fix it
    def optimize_mix_coef(self, data, mix_coef, mean, cov, labels):
        denominator = len(data)
        for k in range(0, len(mix_coef)):
            numerator = 0
            for i in range(0, len(data)):
                numerator += self.zik_indicator(i, k, labels)
            mix_coef[k] = numerator/denominator

    #assigns the point to a cluster
    def assign_cluster(self, point, mean, cov, mix_coef, lam):
        closest = float('-inf')
        decision = 0
        for i in range(0, len(mean)):
            x = self.r_score(point, mean, cov, mix_coef, i, lam)
            if((x)>closest):
                closest = x
                decision = i
        return decision

    #takes the r score of a point where index is the k of r(x,k)
    def r_score(self, point, mean, cov, mix_coef, index, lam):
        nominator = self.weighted_normal_function(point, mean[index], lam[index]*cov[index], mix_coef[index])
        denominator = 0
        for j in range(0, len(mean)):
            denominator += self.weighted_normal_function(point, mean[j], lam[j]*cov[j], mix_coef[j])
        return nominator/denominator

    def log_likelihood(self, data, mean, cov, lam, mix_coef, labels):
        log_l = 0
        log_l2 = 0
        log_temp = 1 #for numerical stability
        for k in range(0, len(mix_coef)):
            log_l += len(self.data[0])*self.calculate_n_k(data, labels, k)*np.log(lam[k])
        for k in range(0, len(mix_coef)):
            for i in range(0, len(self.data)):
                log_l2 += 1/lam[k]*self.zik_indicator(i, k, labels)*np.dot(np.subtract(self.data[i], mean[k])[np.newaxis, :].T.T,np.dot((np.linalg.inv(cov[k])), np.subtract(self.data[i], mean[k]))[np.newaxis, :].T).item()

        return log_l+log_l2

    def calculate_n_k(self, data, labels, k):
        n_k = 0
        for i in range(0, len(data)):
            n_k += self.zik_indicator(i, k, labels)
        return n_k

    #indicator
    #returns 1 if point i belongs to cluster k, 0 if not
    def zik_indicator(self, i, k, labels):
        if (labels[i] == k):
            return 1
        else:
            return 0

    #Calculates the weighted norm
    #point is n*1, mean is n*1, cov is n*n, mix_coef is a number
    def weighted_normal_function(self, point, mean, cov, mix_coef):
        #return mix_coef*(2*np.pi)**(-len(cov)/2)*np.linalg.det(cov)**(-1/2)*np.exp((-1/2)*(np.subtract(point, mean)).T*np.linalg.inv(cov)*(np.subtract(point, mean)))
        return mix_coef*(2*np.pi)**(-len(cov)/2)*np.linalg.det(cov)**(-1/2)*np.exp((-1/2)*np.dot((np.subtract(point, mean).T), np.dot(np.linalg.inv(cov),(np.subtract(point, mean)))))



    def init_mean(self, data):
        data = np.array(data)
        numerater = np.zeros(data[0].shape)
        denominator = len(data)

        for i in data:
            numerater = np.add(numerater, i)
        return numerater/denominator

    def load_data(self, path):
        dataset = np.genfromtxt(path, delimiter='\t')
        return dataset


#x = GMM('Data.tsv', 3)
#labels = x.GMM()
#with open('GMM_output.tsv', 'w', newline='') as f_output:
#    tsv_output = csv.writer(f_output, delimiter='\t')
#    tsv_output.writerow(labels)
