import argparse
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

from torch_utils import get_loaders_objectnet

np.set_printoptions(threshold=np.inf)

model_names = set(filename.split('.')[0].replace('_pca', '') for filename in os.listdir('./results'))

parser = argparse.ArgumentParser(description='IM')
parser.add_argument('--model', dest='model', type=str, default='resnext152_infomin',
                    help='Model: one of' + ', '.join(model_names))
parser.add_argument('--over', type=float, default=1., help='Mutiplier for number of clusters')
parser.add_argument('--n-components', type=int, default=None, help='Number of components for PCA')
args = parser.parse_args()
print(args)

n_classes = 1000
n_clusters = int(args.over * n_classes)
n_classes_objectnet = 313
n_clusters_objectnet = int(args.over * n_classes_objectnet)
train_size = 12811  # 67
val_size = 500  # 00

epochs = 60

n_features = 2048
batch_size = max(2048, int(2 ** np.ceil(np.log2(n_clusters))))


def get_cost_matrix(y_pred, y, nc=1000):
    C = np.zeros((nc, y.max() + 1))
    for pred, label in zip(y_pred, y):
        C[pred, label] += 1
    return C


def get_cost_matrix_objectnet(y_pred, y, objectnet_to_imagenet):
    C = np.zeros((n_clusters, y.max() + 1))
    ny, nyp = [], []
    for pred, label in zip(y_pred, y):
        if len(objectnet_to_imagenet[label]) > 0:
            C[pred, label] += 1
            ny.append(label)
            nyp.append(pred)
    return C, np.array(nyp), np.array(ny)


def get_best_clusters(C, k=3):
    Cpart = C / (C.sum(axis=1, keepdims=True) + 1e-5)
    Cpart[C.sum(axis=1) < 10, :] = 0
    # print('as', np.argsort(Cpart, axis=None)[::-1])
    ind = np.unravel_index(np.argsort(Cpart, axis=None)[::-1], Cpart.shape)[0]  # indices of good clusters
    _, idx = np.unique(ind, return_index=True)
    cluster_idx = ind[np.sort(idx)]  # unique indices of good clusters
    accs = Cpart.max(axis=1)[cluster_idx]
    good_clusters = cluster_idx[:k]
    print('Best clusters accuracy: {}'.format(Cpart[good_clusters].max(axis=1)))
    print('Best clusters classes: {}'.format(Cpart[good_clusters].argmax(axis=1)))
    return good_clusters


def get_worst_clusters(C, k=3):
    Cpart = C / (C.sum(axis=1, keepdims=True) + 1e-5)
    Cstd = Cpart.std(axis=1)
    Cstd[C.sum(axis=1) < 10] = 1000
    cluster_idx = np.argsort(Cstd)  # low std -- closer to uniform
    return cluster_idx[:k]


def print_cluster(ci, y_pred, text):
    idx = np.where(y_pred == ci)[0]
    print('{}: {}'.format(text, idx))


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

    # best matching class for every cluster
    mask = np.ones(C.shape[0], dtype=bool)
    mask[row_ind] = False

    return row_ind.astype(int), col_ind.astype(int)


def imagenet_assignment_to_objectnet(row_ind, col_ind, imagenet_to_objectnet):
    nri, nci = [], []
    for i, (ri, ci) in enumerate(zip(row_ind, col_ind)):
        if imagenet_to_objectnet[ci] > 0:
            nri.append(ri)
            nci.append(imagenet_to_objectnet[ci])
    return np.array(nri), np.array(nci)


def accuracy_from_assignment(C, row_ind, col_ind, set_size=None):
    if set_size is None:
        set_size = C.sum()
    cnt = C[row_ind, col_ind].sum()
    return cnt / set_size


def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def print_metrics(message, y_pred, y_true, train_lin_assignment, train_maj_assignment, val_lin_assignment=None,
                  val_maj_assignment=None, objectnet=False, imagenet_to_objectnet=None, objectnet_to_imagenet=None):
    if objectnet:
        C, y_pred, y_true = get_cost_matrix_objectnet(y_pred, y_true, objectnet_to_imagenet)
        train_lin_assignment = imagenet_assignment_to_objectnet(*train_lin_assignment, imagenet_to_objectnet)
        train_maj_assignment = imagenet_assignment_to_objectnet(*train_maj_assignment, imagenet_to_objectnet)
    else:
        C = get_cost_matrix(y_pred, y_true, n_clusters)

    # for r,c in zip(*train_lin_assignment):
    #     print(r,c)
    acc_tr_lin = accuracy_from_assignment(C, *train_lin_assignment)
    acc_tr_maj = accuracy_from_assignment(C, *train_maj_assignment)
    if val_lin_assignment is not None:
        acc_va_lin = accuracy_from_assignment(C, *val_lin_assignment)
        acc_va_maj = accuracy_from_assignment(C, *val_maj_assignment)
    else:
        acc_va_lin, acc_va_maj = 0, 0

    # confusion_mat(C, *train_lin_assignment, name=args.model)

    ari = sklearn.metrics.adjusted_rand_score(y_true, y_pred)
    v_measure = sklearn.metrics.v_measure_score(y_true, y_pred)
    ami = sklearn.metrics.adjusted_mutual_info_score(y_true, y_pred)
    fm = sklearn.metrics.fowlkes_mallows_score(y_true, y_pred)

    print("{}: ARI {:.5e}\tV {:.5e}\tAMI {:.5e}\tFM {:.5e}".format(message, ari, v_measure, ami, fm))
    print("{}: ACC TR L {:.5e}\tACC TR M {:.5e}\t"
          "ACC VA L {:.5e}\tACC VA M {:.5e}".format(message, acc_tr_lin, acc_tr_maj, acc_va_lin, acc_va_maj))

    if message == 'ont':
        ri, ci = train_lin_assignment
        both = np.zeros(len(ci), dtype=bool)
        y = [s for s in objectnet_to_imagenet if len(objectnet_to_imagenet[s]) > 0]
        for i in range(len(ci)):
            if ci[i] in y:
                both[i] = 1

        acc_both = accuracy_from_assignment(C, ri[both], ci[both], C[:, ci[both]].sum())
        acc_obj = accuracy_from_assignment(C, ri[~both], ci[~both], C[:, ci[~both]].sum())
        print("{}: ACC both {:.5e}\tACC obj {:.5e}".format(message, acc_both, acc_obj))

    best = get_best_clusters(C, k=3)
    worst = get_worst_clusters(C, k=3)
    return best, worst


def train_pca(X_train):
    bs = max(4096, X_train.shape[1] * 2)
    transformer = IncrementalPCA(batch_size=bs)  #
    for i, batch in enumerate(tqdm(batches(X_train, bs), total=len(X_train) // bs + 1)):
        transformer = transformer.partial_fit(batch)
        # break
    print(transformer.explained_variance_ratio_.cumsum())
    return transformer


def cluster_data(X_train, y_train, X_test, y_test, X_test2, y_test2, imagenet_to_objectnet, objectnet_to_imagenet):
    minib_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_no_improvement=None)

    # TODO: save to csv
    for e in trange(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for batch in batches(X_train, batch_size):
            minib_k_means = minib_k_means.partial_fit(batch)

        pred = minib_k_means.predict(X_train)
        C_train = get_cost_matrix(pred, y_train, n_clusters)

        y_pred = minib_k_means.predict(X_test)
        C_val = get_cost_matrix(y_pred, y_test, n_clusters)

        y_pred2 = minib_k_means.predict(X_test2)
        C_val2, _, _ = get_cost_matrix_objectnet(y_pred2, y_test2, objectnet_to_imagenet)

        best_im, worst_im = print_metrics('val', y_pred, y_test, assign_classes_hungarian(C_train),
                                          assign_classes_majority(C_train), assign_classes_hungarian(C_val),
                                          assign_classes_majority(C_val))
        best_obj, worst_obj = print_metrics('on', y_pred2, y_test2, assign_classes_hungarian(C_train),
                                            assign_classes_majority(C_train), assign_classes_hungarian(C_val2),
                                            assign_classes_majority(C_val2), objectnet=True,
                                            imagenet_to_objectnet=imagenet_to_objectnet,
                                            objectnet_to_imagenet=objectnet_to_imagenet)

        for i, cli in enumerate(best_im):
            print_cluster(cli, y_pred, 'best imagenet cluster #{}, imagenet index:'.format(i))
            print_cluster(cli, y_pred2, 'best imagenet cluster #{}, objectnet index:'.format(i))
        for i, cli in enumerate(worst_im):
            print_cluster(cli, y_pred, 'worst imagenet cluster #{}, imagenet index:'.format(i))
            print_cluster(cli, y_pred2, 'worst imagenet cluster #{}, objectnet index:'.format(i))
        if False:
            for i, cli in enumerate(best_obj):
                print_cluster(cli, y_pred, 'best objectnet cluster #{}, imagenet index:'.format(i))
                print_cluster(cli, y_pred2, 'best objectnet cluster #{}, objectnet index:'.format(i))
            for i, cli in enumerate(worst_obj):
                print_cluster(cli, y_pred, 'worst objectnet cluster #{}, imagenet index:'.format(i))
                print_cluster(cli, y_pred2, 'worst objectnet cluster #{}, objectnet index:'.format(i))


def cluster_training_data(X_train, y_train, objectnet_to_imagenet):
    minib_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters_objectnet, batch_size=batch_size,
                                            max_no_improvement=None)

    for e in trange(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for batch in batches(X_train, batch_size):
            minib_k_means = minib_k_means.partial_fit(batch)

        pred = minib_k_means.predict(X_train)
        C_train = get_cost_matrix(pred, y_train, nc=n_clusters_objectnet)

        print_metrics('ont', pred, y_train, assign_classes_hungarian(C_train), assign_classes_majority(C_train),
                      objectnet_to_imagenet=objectnet_to_imagenet)


def transform_pca(X, transformer):
    n = max(4096, X.shape[1] * 2)
    for i in trange(0, len(X), n):
        X[i:i + n] = transformer.transform(X[i:i + n])
        # break
    return X


generate = False
if generate:
    t0 = time.time()
    X, y = make_blobs(n_samples=(train_size + val_size), centers=n_classes, n_features=n_features,
                      center_box=(-0.5, 0.5), random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_size, random_state=42)
    t1 = time.time()

    print('Generation time: {:.6f}'.format(t1 - t0))
    cluster_data(X_train, y_train, X_test, y_test)
else:
    filename = 'results/' + args.model + '_pca.npz'
    if not os.path.exists(filename):

        t0 = time.time()
        path = 'results/' + args.model + '.npz'
        data = np.load(path)
        X_train, y_train, X_test, y_test, X_test2, y_test2 = data['train_embs'], data['train_labs'], data['val_embs'], \
                                                             data['val_labs'], data['obj_embs'], data['obj_labs']
        t1 = time.time()
        print(path)
        print('Loading time: {:.6f}'.format(t1 - t0))

        if len(y_train.shape) > 1:
            y_train, y_test, y_test2 = y_train.argmax(1), y_test.argmax(1), y_test2.argmax(1)
        transformer = train_pca(X_train)
        X_train, X_test = transform_pca(X_train, transformer), transform_pca(X_test, transformer)
        gc.collect()
        np.savez(filename, train_embs=X_train, train_labs=y_train, val_embs=X_test, val_labs=y_test, obj_embs=X_test2,
                 obj_labs=y_test2, PCA=transformer)
    else:
        t0 = time.time()
        data = np.load(filename)
        print(filename)
        X_train, y_train, X_test, y_test, X_test2, y_test2 = data['train_embs'], data['train_labs'], data['val_embs'], \
                                                             data['val_labs'], data['obj_embs'], data['obj_labs']
        # print(y_test2.shape, y_test2, y_test2.max())
        if len(y_test2.shape) > 1:
            y_test2 = y_test2.argmax(1)
        t1 = time.time()
        print('Loading time: {:.6f}'.format(t1 - t0))
    if args.n_components is not None:
        X_train, X_test, X_test2 = X_train[:, :args.n_components], X_test[:, :args.n_components], X_test2[:,
                                                                                                  :args.n_components]

    objectnet_path = '/home/chaimb/objectnet-1.0'
    imagenet_path = '/home/chaimb/ILSVRC/Data/CLS-LOC'
    val_loader, imagenet_to_objectnet, objectnet_to_imagenet, objectnet_both, imagenet_both = get_loaders_objectnet(
        objectnet_path, imagenet_path, 16, 224, 8, 1, 0)
    cluster_data(X_train, y_train, X_test, y_test, X_test2, y_test2, imagenet_to_objectnet, objectnet_to_imagenet)
    cluster_training_data(X_test2, y_test2, objectnet_to_imagenet)
