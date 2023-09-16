import os
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
import warnings
import pandas as pd

warnings.filterwarnings("ignore")  # This will ignore all the warnings;
from sklearn.metrics import accuracy_score

# K for KNN: [2, 4, 6, 8, 10, 16, 32, 64, 128]
# TODO: adjust the range

# K-Flod cross validation
K = 5

# hyperparameter k for neighbours
k_list = [3, 5, 7, 11, 15, 19, 25, 55]


class myKNN(object):
    def __init__(self, n_neighbors, weights, metric, X_train, X_test, y_train, y_test):
        self.k = n_neighbors
        self.metric = metric
        self.weights = weights
        self.model = KNN(n_neighbors=n_neighbors, weights=weights, metric=metric)  # ad
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.ACC = 0
        self.runtime = 0

    def model_train_and_test(self):
        st = time.time()
        self.model.fit(self.X_train, self.y_train.ravel())
        et = time.time()

        print("-----------KNN use %.4f s to train.-----------" % (et - st))
        st = time.time()
        self.ACC = accuracy_score(self.y_test, self.model.predict(self.X_test))
        et = time.time()
        self.runtime = et - st
        print("-----------KNN use %.4f s to test.-----------" % self.runtime)
        print("Parameter K:{}".format(self.k))
        print("Distance Metric:{}".format(self.metric))
        print("Distance Weights:{}".format(self.weights))
        print("Test set score:{:.4f}".format(self.ACC))

        return self.ACC, self.runtime
        # print("----------------------------------------------------")

    def parameter_search(self):
        para_search = GridSearchCV(self.model, {"K": k_list}, cv=K, verbose=10, n_jobs=K)
        print("type:", type(self.y_train))  # , "shape:",self.y_train.sha)
        st = time.time()
        para_search.fit(self.X_train, self.y_train.ravel())
        et = time.time()
        print("--------KNN use %.4f s to fit.--------" % (et - st))

        print("Parameter K list:{}".format(k_list))
        print("Best K:{}".format(para_search.best_params_))
        print("Best score (mean ACC) on training set:{:.4f}".format(para_search.best_score_))
        print("Test set score:{:.4f}".format(para_search.score(self.X_test, self.y_test)))
