import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings("ignore") # This will ignore all the warnings;

# C for SVM: [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
# TODO: adjust the range
# C_list = [pow(10, i) for i in range(-2, 6)]
# C_list = [0.05, 0.1, 0.5, 1, 5, 10, 100]
C_list = [0.0001, 0.001, 0.05, 0.1, 0.5, 1, 10]
# K-Flod cross validation
K = 5


class LinearSVM(object):
    def __init__(self, features, labels, dim):
        self.model = LinearSVC(max_iter=2000) # adjusted max_iter
        self.dim = dim
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features,
                                                                                labels,
                                                                                test_size=0.4,
                                                                                random_state=42,
                                                                                stratify=labels)


    def parameter_search(self):
        para_search = GridSearchCV(self.model, {"C": C_list}, cv=K, verbose=10, n_jobs=K)
        print("type:", type(self.y_train))#, "shape:",self.y_train.sha)
        para_search.fit(self.X_train, self.y_train.ravel())

        print("Dim:{}".format(self.dim))
        print("Parameter C list:{}".format(C_list))
        print("Best C:{}".format(para_search.best_params_))
        print("Best score (mean ACC) on training set:{:.4f}".format(para_search.best_score_))
        print("Test set score:{:.4f}".format(para_search.score(self.X_test, self.y_test)))