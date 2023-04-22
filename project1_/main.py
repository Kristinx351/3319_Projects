import numpy as np
import os
from SVM import LinearSVM
import time
from project1.DR_methods.FP_methods import FP_LDA
from project1.DR_methods.FS_methods import FS_RFE
from project1.DR_methods.FL_methods import FL_TSNE, FL_LLE
import warnings
warnings.filterwarnings("ignore") # This will ignore all the warnings;

path = r'E:\Study_0\Term6\Data_Science\Projects\project\AwA2-features\ResNet101'
# Trying dims : 8, 16, 32, 64, 128, 256, 512, 1024, 2048
# dim_list = [pow(2, i) for i in range(3, 12)]
dim_list = [16]  # for test
print("Trying dimension:", dim_list)
# method_list = ["PCA", "LDA", "RFE", "TSNE", "LLE"]
# test
method_list = ["RFE"]

# FP = method_list[:2]
# FS = method_list[2]
# FL = method_list[3:]

FP = ["PCA", "LDA"]
FS = ["RFE"]
FL = ["TSNE", "LLE"]
runtime = np.array((len(method_list), len(dim_list)))


def generate_features(m):
    feature = np.load(os.path.join(path, 'features.npy'))
    label = np.load(os.path.join(path, 'labels.npy'))
    if m in FP:
        by_FP_methods(m,  feature, label)
    elif m in FS:
        by_FS_methods(m, feature, label)
    else:
        by_FL_methods(m, feature, label)


def by_FS_methods(m, feature, label):
    for dim in dim_list:
        FS_RFE(feature, label, dim)
        print(
            "-----Successfully generate feature files of dim %d by %d feature selection methods!-----" % (dim, len(FS)))


def by_FP_methods(m, feature, label):
    for dim in dim_list:
        #FP_PCA(feature, dim)
        FP_LDA(feature, dim, label)

        print("-----Successfully generate feature files of dim %d by %d feature projection methods!-----" % (dim, len(FP)))


def by_FL_methods(m, feature, label):
    if m == "TSNE":
        FL_TSNE(feature, 3)
    else:
        for dim in dim_list:
            FL_LLE(feature, dim, label)

        print(
            "-----Successfully generate feature files of dim %d by %d feature learning methods!-----" % (dim, len(FL)))


if __name__ == "__main__":
    for m in method_list:

        # Run it at the first time to generate corresponding .npy files.
        # generate_features(m)

        for dim in dim_list:
            start_time = time.time()

            if dim == 2048:
                features = np.load(os.path.join(path, 'features.npy'))
            if m == "TSNE":
                features = np.load(os.path.join(path, r'{}\features_{}.npy'.format(m, 3)))
            else:
                features = np.load(os.path.join(path, r'{}\features_{}.npy'.format(m, dim)))
            labels = np.load(os.path.join(path, 'labels.npy'))
            classifier = LinearSVM(features, labels, dim)
            classifier.parameter_search()

            end_time = time.time()
            print("###############################################################")
            print("Spent %.4f s to classify [%s method w %d dim feature ]. " % (end_time - start_time, m, dim))
            print("###############################################################")
