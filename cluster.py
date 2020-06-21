import os
import time

import numpy as np
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn import cluster
from sklearn.datasets import make_blobs
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import trange, tqdm

n_classes = 1000
n_clusters = 2 * n_classes  # https://arxiv.org/abs/2005.12320 -- overclustering
train_size = 12811  # 67
val_size = 500  # 00

epochs = 60

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


def assign_classes_hungarian(C):
    row_ind, col_ind = linear_sum_assignment(C, maximize=True)
    ri, ci = np.arange(C.shape[0]), np.zeros(C.shape[0])
    ci[row_ind] = col_ind

    # for overclustering, rest is assigned to best matching class
    mask = np.ones(C.shape[0], dtype=bool)
    mask[row_ind] = False
    ci[mask] = C[mask, :].argmax(1)
    return ri.astype(int), ci.astype(int)


def assign_classes_majority(C):
    col_ind = C.argmax(1)
    row_ind = np.arange(C.shape[0])

    # for overclustering, rest is assigned to best matching class
    mask = np.ones(C.shape[0], dtype=bool)
    mask[row_ind] = False

    return row_ind.astype(int), col_ind.astype(int)


def accuracy_from_assignment(C, row_ind, col_ind, set_size):
    cnt = C[row_ind, col_ind].sum()
    return cnt / set_size


def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def print_metrics(y_pred, y_true, train_lin_assignment, train_maj_assignment, val_lin_assignment, val_maj_assignment):
    C = get_cost_matrix(y_pred, y_true)

    acc_tr_lin = accuracy_from_assignment(C, *train_lin_assignment, len(y_true))
    acc_tr_maj = accuracy_from_assignment(C, *train_maj_assignment, len(y_true))
    acc_va_lin = accuracy_from_assignment(C, *val_lin_assignment, len(y_true))
    acc_va_maj = accuracy_from_assignment(C, *val_maj_assignment, len(y_true))

    ari = sklearn.metrics.adjusted_rand_score(y_true, y_pred)
    v_measure = sklearn.metrics.v_measure_score(y_true, y_pred)
    ami = sklearn.metrics.adjusted_mutual_info_score(y_true, y_pred)
    fm = sklearn.metrics.fowlkes_mallows_score(y_true, y_pred)

    print("ARI {:.4f}\tV {:.4f}\tAMI {:.4f}\tFM {:.4f}\n".format(ari, v_measure, ami, fm))
    print("ACC TR L {:.4f}\tACC TR M {:.4f}\t"
          "ACC VA L {:.4f}\tACC VA M {:.4f}\n".format(acc_tr_lin, acc_tr_maj, acc_va_lin, acc_va_maj))


def train_pca(n_components=128):
    transformer = IncrementalPCA(n_components=n_components, batch_size=4096)  #
    for i, batch in enumerate(tqdm(batches(X_train, 4096), total=len(X_train) // 4096)):
        transformer = transformer.partial_fit(batch)
    print(transformer.explained_variance_ratio_.cumsum())
    return transformer


def cluster_data(X_train, y_train, X_test, y_test):
    mean, std = X_train.mean(0), X_train.std()
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    minib_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_no_improvement=None)

    for e in trange(epochs):
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        for batch in batches(X_train, batch_size):
            minib_k_means = minib_k_means.partial_fit(batch)

        pred = minib_k_means.predict(X_train)
        C_train = get_cost_matrix(pred, y_test)

        y_pred = minib_k_means.predict(X_test)
        C_val = get_cost_matrix(y_pred, y_test)

        print_metrics(y_pred, y_test, assign_classes_hungarian(C_train), assign_classes_majority(C_train),
                      assign_classes_hungarian(C_val), assign_classes_majority(C_val))


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
    n_components = 128
    filename = 'results/resnet50_pca{}.npz'.format(n_components)
    if not os.path.exists(filename):

        t0 = time.time()
        path = 'results/resnet50.npz'
        data = np.load(path)
        X_train, y_train, X_test, y_test = data['train_embs'], data['train_labs'], data['val_embs'], data['val_labs']
        t1 = time.time()

        print('Loading time: {:.4f}'.format(t1 - t0))
        transformer = train_pca(n_components=128)
        X_train, X_test = transformer.transform(X_train), transformer.transform(X_test)
        np.savez(filename, train_embs=X_train, train_labs=y_train, val_embs=X_test, val_labs=y_test)
    else:
        data = np.load(filename)
        X_train, y_train, X_test, y_test = data['train_embs'], data['train_labs'], data['val_embs'], data['val_labs']
    cluster_data(X_train, y_train, X_test, y_test)

# k-means: 0.0370
# PCA+k-means: 0.223
