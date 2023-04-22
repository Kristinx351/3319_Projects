import numpy as np
import os
import time


class BaseModel(object):
    def __init__(self, feature, dim, method):
        self.method = method
        self.dim = dim
        self.model = 'Base'
        self.start_time = time.time()
        self.feature = feature
        self.end_time = time.time()
        self.generate_time = self.end_time - self.start_time



