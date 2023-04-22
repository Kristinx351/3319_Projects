from sklearn.decomposition import PCA
import numpy as np
import os
from project1.DR_methods.base import BaseModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import time

path = r'E:\Study_0\Term6\Data_Science\Projects\project\AwA2-features\ResNet101'


class FP_PCA(BaseModel):
    def __init__(self, feature, dim, method="PCA"):
        super().__init__(feature, dim, method)
        self.start_time = time.time()
        self.model = PCA(n_components=dim)
        self.feature = self.model.fit_transform(feature)
        self.end_time = time.time()
        self.generate_time = self.end_time - self.start_time
        print("---%s method used %.4f to generate %d dim feature.---" % (self.method, self.generate_time, dim))

        np.save(os.path.join(path, r'{}\features_{}.npy'.format(self.method, dim)), self.feature)


class FP_LDA(BaseModel):
    def __init__(self, feature, dim, label, method="LDA"):
        super().__init__(feature, dim, method)
        self.start_time = time.time()
        self.model = LDA(n_components=dim)
        self.label = label
        self.feature = self.model.fit_transform(feature, label)
        self.end_time = time.time()
        self.generate_time = self.end_time - self.start_time
        print("---%s method used %.4f to generate %d dim feature.---" % (self.method, self.generate_time, dim))

        np.save(os.path.join(path, r'{}\features_{}.npy'.format(self.method, dim)), self.feature)
