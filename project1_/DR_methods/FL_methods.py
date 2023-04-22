import numpy as np
import os
from project1.DR_methods.base import BaseModel
from sklearn.manifold import TSNE, locally_linear_embedding as LLE
import time

path = r'E:\Study_0\Term6\Data_Science\Projects\project\AwA2-features\ResNet101'


class FL_TSNE(BaseModel):
    def __init__(self, feature, dim, method="TSNE"):
        super().__init__(feature, dim, method)
        self.start_time = time.time()
        self.model = TSNE(n_components=dim)
        self.feature = self.model.fit_transform(feature)
        self.end_time = time.time()
        self.generate_time = self.end_time - self.start_time
        print("---%s method used %.4fs to generate %d dim feature.---" % (self.method, self.generate_time, dim))

        np.save(os.path.join(path, r'{}\features_{}.npy'.format(self.method, dim)), self.feature)

class FL_LLE(BaseModel):
    def __init__(self, feature, dim, method="LLE"):
        super().__init__(feature, dim, method)
        self.start_time = time.time()
        self.model = LLE(feature, n_neighbors= 5, n_components=dim)
        self.feature = self.model.fit_transform(feature)
        self.end_time = time.time()
        self.generate_time = self.end_time - self.start_time
        print("---%s method used %.4fs to generate %d dim feature.---" % (self.method, self.generate_time, dim))

        np.save(os.path.join(path, r'{}\features_{}.npy'.format(self.method, dim)), self.feature)
