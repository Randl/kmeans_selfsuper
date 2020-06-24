import gc
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
n_clusters = 1 * n_classes  # https://arxiv.org/abs/2005.12320 -- overclustering
train_size = 12811  # 67
val_size = 500  # 00

epochs = 60

n_features = 2048
batch_size = max(2048, int(2 ** np.ceil(np.log2(n_clusters))))


def get_cost_matrix(y_pred, y):
    C = np.zeros((n_clusters, n_classes))
    for pred, label in zip(y_pred, y):
        C[pred, label] += 1
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


def print_metrics(y_pred, y_true,
                  train_lin_assignment, train_maj_assignment,
                  val_lin_assignment, val_maj_assignment):
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
          "ACC VA L {:.4f}\tACC VA M {:.4f}\n".format(acc_tr_lin, acc_tr_maj,
                                                      acc_va_lin, acc_va_maj))


def train_pca(X_train):
    bs = max(4096, X_train.shape[1] * 2)
    transformer = IncrementalPCA(batch_size=bs)  #
    for i, batch in enumerate(tqdm(batches(X_train, bs), total=len(X_train) // bs)):
        transformer = transformer.partial_fit(batch)
    print(transformer.explained_variance_ratio_.cumsum())
    return transformer


def cluster_data(X_train, y_train, X_test, y_test):
    minib_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_no_improvement=None)

    for e in trange(epochs):
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        for batch in batches(X_train, batch_size):
            minib_k_means = minib_k_means.partial_fit(batch)

        pred = minib_k_means.predict(X_train)
        C_train = get_cost_matrix(pred, y_train)

        y_pred = minib_k_means.predict(X_test)
        C_val = get_cost_matrix(y_pred, y_test)

        print_metrics(y_pred, y_test, assign_classes_hungarian(C_train), assign_classes_majority(C_train),
                      assign_classes_hungarian(C_val), assign_classes_majority(C_val))


def transform_pca(X, transformer):
    n = max(4096, X.shape[1] * 2)
    for i in trange(0, len(X), n):
        X[i:i + n] = transformer.transform(X[i:i + n])
        break
    return X


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

        y_train, y_test = y_train.argmax(1), y_test.argmax(1)
        transformer = train_pca(X_train)
        X_train, X_test = transform_pca(X_train, transformer), transform_pca(X_test, transformer)
        gc.collect()
        np.savez(filename, train_embs=X_train, train_labs=y_train, val_embs=X_test, val_labs=y_test)
    else:
        t0 = time.time()
        data = np.load(filename)
        X_train, y_train, X_test, y_test = data['train_embs'], data['train_labs'], data['val_embs'], data['val_labs']
        if len(y_train.shape) > 1:
            y_train, y_test = y_train.argmax(1), y_test.argmax(1)
        t1 = time.time()
        print('Loading time: {:.4f}'.format(t1 - t0))
        X_train, X_test = X_train[:, :n_components], X_test[:, :n_components]
        cluster_data(X_train, y_train, X_test, y_test)

# ResNet50
# k-means: 0.0370
# PCA+k-means: 0.223
# over (2): ARI 0.0936	V 0.6336	AMI 0.2520	FM 0.0969
#
# ACC TR L 0.2293	ACC TR M 0.2386	ACC VA L 0.2714	ACC VA M 0.2774
