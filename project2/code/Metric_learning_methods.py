import numpy as np
import os
import time
from metric_learn import NCA, LFDA, LMNN

class BaseModel(object):
    def __init__(self, X_train, X_test, y_train, y_test, ori_dim, dim, method="Base"):
        self.method = method
        self.ori_dim = ori_dim
        self.dim = dim
        # print(dim)
        # print(type(self.dim))
        # TODO MAX_iter adjust
        self.max_iter = 100
        self.model = 'Base'
        # self.X_train = X_train.astype(np.float16)
        # self.X_test = X_test.astype(np.float16)
        # self.y_train = y_train.astype(np.float16)
        # self.y_test = y_test.astype(np.float16)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # self.start_time = time.time()
        # self.end_time = time.time()
        # self.generate_time = self.end_time - self.start_time


    def save_test_data(self):
        id = f"{self.method}_{self.ori_dim}_{self.dim}"
        save_path = f"project2/Data/{self.method}/{id}"
        os.makedirs(save_path)
        return save_path


class myNCA(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test, ori_dim, dim, method="NCA"):
        super().__init__(X_train, X_test, y_train, y_test, ori_dim, dim, method)
        nca = NCA(n_components=self.dim, max_iter=self.max_iter, verbose=1)
        nca.fit(self.X_train, self.y_train)
        print("NCA fit DONE!")
        self.max_iter = 80
        self.path = self.save_test_data()

        np.save(os.path.join(self.path, "train_feature.npy"), nca.transform(self.X_train))
        np.save(os.path.join(self.path, "test_feature.npy"), nca.transform(self.X_test))


class myLMNN(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test,  ori_dim, dim, method="LMNN"):
        super().__init__(X_train, X_test, y_train, y_test,  ori_dim, dim, method)

        self.method = "LMNN"
        self.max_iter = 150
        self.k = 7
        lmnn = LMNN(k=self.k, n_components=dim, max_iter=self.max_iter, verbose=1, learn_rate=1e-6)

        lmnn.fit(self.X_train.astype(np.float16), self.y_train.astype(np.float16))
        print("LMNN fit DONE!")
        self.path = self.save_test_data()

        np.save(os.path.join(self.path, "train_feature.npy"), lmnn.transform(self.X_train))
        np.save(os.path.join(self.path, "test_feature.npy"), lmnn.transform(self.X_test))


class myLFDA(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test, ori_dim, dim, method="LFDA"):
        super().__init__(X_train, X_test, y_train, y_test, ori_dim, dim, method)
        self.k = 7
        self.method = "LFDA"
        lfda = LFDA(n_components=self.dim, k=self.k)
        lfda.fit(self.X_train, self.y_train)
        print("LFDA fit DONE!")
        self.path = self.save_test_data()

        np.save(os.path.join(self.path, "train_feature.npy"), lfda.transform(self.X_train))
        np.save(os.path.join(self.path, "test_feature.npy"), lfda.transform(self.X_test))




