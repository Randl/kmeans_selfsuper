import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import cluster
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import trange

n_classes = 1000
n_clusters = 1 * n_classes  # https://arxiv.org/abs/2005.12320 -- overclustering
train_size = 12811  # 67
val_size = 500  # 00

epochs = 90

n_features = 2048
batch_size = 2048


def get_cost_matrix(y_pred, y):
    # TODO test
    C1 = np.zeros((n_clusters, y_pred.size))
    C1[y_pred, np.arange(y_pred.size)] = 1
    if len(y.shape) == 1:
        C2 = np.zeros((y.size, n_classes))
        C2[np.arange(y.size), y] = 1
    else:
        C2 = y.astype(int)

    C = np.matmul(C1.astype(int), C2.astype(int))  # num_clusters * num_classes
    return C


def assign_classes(C):
    row_ind, col_ind = linear_sum_assignment(C, maximize=True)
    acc_linear = C[row_ind, col_ind].sum()
    acc_max = C.max(1).sum()
    return acc_linear, acc_max


def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def cluster_data(X_train, y_train, X_test, y_test):
    minib_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_no_improvement=None)

    for e in trange(epochs):
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        for batch in batches(X_train, batch_size):
            minib_k_means = minib_k_means.partial_fit(batch)
        y_pred = minib_k_means.predict(X_test)
        C = get_cost_matrix(y_pred, y_test)
        acc_linear, acc_max = assign_classes(C)
        print("Epoch[{}/{}]: Acc: {:.4f} or {:.4f}".format(e, epochs, acc_linear / len(y_test), acc_max / len(y_test)))


generate = False
if generate:
    t0 = time.time()
    X, y = make_blobs(n_samples=(train_size + val_size), centers=n_classes, n_features=n_features,
                      center_box=(-0.5, 0.5), random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_size, random_state=42)
    t1 = time.time()

    print('Generation time: {:.4f}'.format(t1 - t0))
    cluster_data(X_train, y_train, X_test, y_test)
else:
    path = 'results/resnet50.npz'
    data = np.load(path)
    X_train, y_train, X_test, y_test = data['train_embs'], data['train_labs'], data['val_embs'], data['val_labs']
    cluster_data(X_train, y_train, X_test, y_test)
