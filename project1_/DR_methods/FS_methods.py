import numpy as np
import os
from project1.DR_methods.base import BaseModel
from sklearn import preprocessing
from sklearn.feature_selection import RFE
# from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import time

path = r'E:\Study_0\Term6\Data_Science\Projects\project\AwA2-features\ResNet101'


class FS_RFE(BaseModel):
    def __init__(self, feature, label, dim, method="RFE"):
        super().__init__(feature, dim, method)
        self.start_time = time.time()
        feature = preprocessing.normalize(feature, norm='l1')
        self.model = RFE(LogisticRegression(), n_features_to_select= dim, step=128*2, verbose=1)
        self.model.fit(feature, label)
        # self.model.show()
        self.feature = self.get_features()
        self.end_time = time.time()
        self.generate_time = self.end_time - self.start_time
        print("---%s method used %.4fs to generate %d dim feature.---" % (self.method, self.generate_time, dim))

        np.save(os.path.join(path, r'{}\features_{}.npy'.format(self.method, dim)), self.feature)

    def get_features(self):
        indexes = [i for i in range(self.feature.shape[1])]
        rank = sorted(zip(map(lambda x: round(x, 4), self.model.ranking_), indexes)) #rfe_estimator_

        rank_list = []
        for i in rank:
            rank_list.append(i[1])
        # print(rank)
        # print(rank_list)

        _feature = np.zeros((self.feature.shape[0], self.dim))
        rank_l = sorted(rank_list[:self.dim])

        j = 0
        for i in rank_l:
            _feature[:, j] = self.feature[:, i]
            j += 1

        return _feature
