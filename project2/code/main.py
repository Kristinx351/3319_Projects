import numpy as np
import os
import time
import warnings
import argparse
from KNN import myKNN
import pandas as pd
from sklearn.model_selection import train_test_split
from Metric_learning_methods import myNCA, myLMNN, myLFDA

warnings.filterwarnings("ignore")  # This will ignore all the warnings;

path = r'E:\Study_0\Term6\Data_Science\Projects\project1\AwA2-features\ResNet101'
distance_list = ['l1', 'l2', 'cosine']# "LFDA", "LMNN", "LFDA"]
metric_learn = ["NCA", "LFDA", "LMNN"]
weights = ['uniform', 'distance']
# k_list = [11, 15, 21]
k_list = [3, 5, 7, 9, 11, 15, 21]


def generate_features(X_train, X_test, y_train, y_test, ori, dim):
    #myNCA(X_train, X_test, y_train, y_test, ori, dim)
    myLMNN(X_train, X_test, y_train, y_test, ori, dim)
    #myLFDA(X_train, X_test, y_train, y_test, ori, dim)


if __name__ == "__main__":
    result_path = r"E:\Study_0\Term6\Data_Science\Projects\project2"
    res = pd.DataFrame(columns=distance_list, index=k_list)
    parser = argparse.ArgumentParser("Test")
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--ori_dim', type=int, default=2048)
    # parser.add_argument('--distance', default='Eu')
    # parser.add_argument('--n-neighbors', type=int, default=7)
    parser.add_argument('--weights', type=str, default='distance')
    args = parser.parse_args()

    features = np.load(os.path.join(path, 'features.npy'))
    if args.ori_dim == 2048:
        features = np.load(os.path.join(path, 'features.npy'))
    else:
        print("################ Trying %d dim feature from PCA ################" % args.ori_dim)
        features = np.load(os.path.join(path, r'{}\features_{}.npy'.format("PCA", args.ori_dim)))
    # features = features.astype(np.float16)
    print(features.shape)
    labels = np.load(os.path.join(path, 'labels.npy'))
    labels = labels.astype(np.float16)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42,
                                                        stratify=labels)
    np.save(os.path.join(path, 'test_label.npy'), y_test)
    # X_train = X_train.astype(np.float16)
    # X_test = X_test.astype(np.float16)
    # y_train = y_train.astype(np.float16)
    # y_test = y_test.astype(np.float16)

    generate_features(X_train, X_test, y_train, y_test, args.ori_dim, args.dim)
    #
    # pd.Series([])
    writer = pd.ExcelWriter(f'E:\\Study_0\\Term6\\Data_Science\\Projects\\project2\\result_{args.ori_dim}_{args.dim}.xlsx')
    excel_header = ['K', 'test_ACC', 'run_Time']

    for dis in distance_list[:]:
        acc = []
        runtime = []

        print("++++Trying %s metric for KNN++++" % dis)
        if dis in metric_learn:
            X_train = np.load(f"project2\\Data\\{dis}\\{dis}_{args.ori_dim}_{args.dim}\\train_feature.npy")
            X_test = np.load(f"project2\\Data\\{dis}\\{dis}_{args.ori_dim}_{args.dim}\\test_feature.npy")
        for k in k_list:
            if dis in metric_learn:
                knn_dis = 'l2'
            else: knn_dis = dis
            classifier = myKNN(k, args.weights, knn_dis, X_train, X_test, y_train, y_test)
            ACC, RT = classifier.model_train_and_test()
            acc.append(ACC)
            runtime.append(RT)
            # classifier.parameter_search()
        # ++++++++++++++++")
        scr = pd.DataFrame(data={'K': k_list, 'test_ACC': acc, 'run_Time': runtime})
        scr.to_excel(writer, sheet_name=f'{dis}_{args.ori_dim}_{args.dim}', header=excel_header, index=False)

    writer.save()

    print("###### Test Done for %d dim feature and %s weights!#####" % (args.dim, args.weights))
