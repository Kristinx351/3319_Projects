import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser("Test")
parser.add_argument('--method', default='LMNN')
parser.add_argument('--ori_dim', type=int, default=128)
parser.add_argument('--dim', type=int, default=16)
parser.add_argument('--N', type=int, default=2)
args = parser.parse_args()

path = r'E:\Study_0\Term6\Data_Science\Projects\project1\AwA2-features\ResNet101'


def plot_2D(y, SNE_X):
    X_tsne_data = np.vstack((SNE_X.T, y.T)).T
    df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class'])

    # x_min, x_max = X.min(0), X.max(0)
    # X_normalized = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2')
    plt.show()
    plt.savefig(os.path.join(r"E:\Study_0\Term6\Data_Science\Projects\project2\Figs",
                             f"{args.N}D_tSNE\{args.method}_{args.ori_dim}_{args.dim}.png"))



def plot_3D(y, SNE_X):
    X_tsne_data = np.vstack((SNE_X.T, y.T)).T
    df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'Dim3', 'class'])

    plt.figure(figsize=(16, 16))
    axes = plt.axes(projection='3d')
    print(type(axes))
    x1 = df_tsne['Dim1']
    y1 = df_tsne['Dim2']
    z1 = df_tsne['Dim3']
    axes.scatter3D(x1, y1, z1, s=85, linewidth=0.2, c=df_tsne['class'], edgecolors='white', alpha=0.65)

    axes.set_xlabel('Dim1')
    axes.set_ylabel('Dim2')
    axes.set_zlabel('Dim3')
    plt.show()
    plt.savefig(os.path.join(r"E:\Study_0\Term6\Data_Science\Projects\project2\Figs",
                             f"{args.N}D_tSNE\{args.method}_{args.ori_dim}_{args.dim}.pdf"))


X = np.load(os.path.join(f'E:\\jupyter_notebook\\DS_projects\\project2\\Data\\{args.method}\\{args.method}_{args.ori_dim}_{args.dim}', 'test_feature.npy'))
y= np.load(os.path.join(path, 'test_label.npy'))

tSNE = TSNE(n_components=args.N, verbose=1, init='pca', method='barnes_hut')
SNE_X = tSNE.fit_transform(X)
print(y.shape, SNE_X.shape)

if args.N == 2:
    plot_2D(y, SNE_X)

if args.N == 3:
    plot_3D(y, SNE_X)

