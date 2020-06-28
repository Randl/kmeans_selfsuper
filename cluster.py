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

import zipfile
import io


def saveCompressed(fh, **namedict):
    print('startsave')
    with zipfile.ZipFile(fh,
                         mode="w",
                         compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zf:
        for k, v in namedict.items():
            buf = io.BytesIO()
            print(type(v))
            np.lib.npyio.format.write_array(buf,
                                            np.asanyarray(v),
                                            allow_pickle=False)
            zf.writestr(k + '.npy',
                        buf.getvalue())
            del v
            gc.collect()


def get_cost_matrix(y_pred, y, nc=1000):
    C = np.zeros((nc, y.max() + 1))
    for pred, label in zip(y_pred, y):
        C[pred, label] += 1
    return C


def get_cost_matrix_objectnet(y_pred, y, objectnet_to_imagenet):
    C = np.zeros((n_clusters, y.max()+1))
    ny, nyp = [], []
    for pred, label in zip(y_pred, y):
        if len(objectnet_to_imagenet[label]) > 0:
            C[pred, label] += 1
            ny.append(label)
            nyp.append(pred)
    return C, np.array(nyp), np.array(ny)


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
        set_size=C.sum()
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
        C = get_cost_matrix(y_pred, y_true)

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

    # TODO: check which classes are not assigned
    print("{}: ARI {:.4f}\tV {:.4f}\tAMI {:.4f}\tFM {:.4f}".format(message, ari, v_measure, ami, fm))
    print("{}: ACC TR L {:.4f}\tACC TR M {:.4f}\t"
          "ACC VA L {:.4f}\tACC VA M {:.4f}".format(message, acc_tr_lin, acc_tr_maj, acc_va_lin, acc_va_maj))

    if message=='ont': #TODO
        both = np.zeros(313, dtype=bool)
        y = [s for s in objectnet_to_imagenet if len(objectnet_to_imagenet[s]) > 0]
        both[y] = 1

        ri, ci = train_lin_assignment
        acc_both = accuracy_from_assignment(C, ri[both], ci[both], C[:, ci[both]].sum())
        acc_obj = accuracy_from_assignment(C, ri[~both], ci[~both], C[:, ci[~both]].sum())
        print("{}: ACC both {:.4f}\tACC obj {:.4f}".format(message, acc_both, acc_obj))


def train_pca(X_train):
    bs = max(4096, X_train.shape[1] * 2)
    transformer = IncrementalPCA(batch_size=bs)  #
    for i, batch in enumerate(tqdm(batches(X_train, bs), total=len(X_train) // bs + 1)):
        transformer = transformer.partial_fit(batch)
    print(transformer.explained_variance_ratio_.cumsum())
    return transformer


def cluster_data(X_train, y_train, X_test, y_test, X_test2, y_test2, imagenet_to_objectnet, objectnet_to_imagenet):
    minib_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_no_improvement=None)

    for e in trange(epochs):
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        for batch in batches(X_train, batch_size):
            minib_k_means = minib_k_means.partial_fit(batch)

        pred = minib_k_means.predict(X_train)
        C_train = get_cost_matrix(pred, y_train)

        y_pred = minib_k_means.predict(X_test)
        C_val = get_cost_matrix(y_pred, y_test)

        y_pred2 = minib_k_means.predict(X_test2)
        C_val2, _, _ = get_cost_matrix_objectnet(y_pred2, y_test2, objectnet_to_imagenet)

        print_metrics('val', y_pred, y_test, assign_classes_hungarian(C_train), assign_classes_majority(C_train),
                      assign_classes_hungarian(C_val), assign_classes_majority(C_val))
        print_metrics('on', y_pred2, y_test2, assign_classes_hungarian(C_train), assign_classes_majority(C_train),
                      assign_classes_hungarian(C_val2), assign_classes_majority(C_val2), objectnet=True,
                      imagenet_to_objectnet=imagenet_to_objectnet, objectnet_to_imagenet=objectnet_to_imagenet)


def cluster_training_data(X_train, y_train,objectnet_to_imagenet):
    minib_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters_objectnet, batch_size=batch_size,
                                            max_no_improvement=None)

    for e in trange(epochs):
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        for batch in batches(X_train, batch_size):
            minib_k_means = minib_k_means.partial_fit(batch)

        pred = minib_k_means.predict(X_train)
        C_train = get_cost_matrix(pred, y_train, nc=313)

        print_metrics('ont', pred, y_train, assign_classes_hungarian(C_train), assign_classes_majority(C_train),objectnet_to_imagenet=objectnet_to_imagenet)


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

    print('Generation time: {:.4f}'.format(t1 - t0))
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
        print('Loading time: {:.4f}'.format(t1 - t0))

        if len(y_train.shape) > 1:
            y_train, y_test = y_train.argmax(1), y_test.argmax(1)
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
        if len(y_train.shape) > 1:
            y_train, y_test = y_train.argmax(1), y_test.argmax(1)
        t1 = time.time()
        print('Loading time: {:.4f}'.format(t1 - t0))
    if args.n_components is not None:
        X_train, X_test, X_test2 = X_train[:, :args.n_components], X_test[:, :args.n_components], X_test2[:,
                                                                                                  :args.n_components]

    objectnet_path = '/home/chaimb/objectnet-1.0'
    imagenet_path = '/home/chaimb/ILSVRC/Data/CLS-LOC'
    val_loader, imagenet_to_objectnet, objectnet_to_imagenet, objectnet_both, imagenet_both = get_loaders_objectnet(
        objectnet_path, imagenet_path, 16, 224, 8, 1, 0)
    cluster_data(X_train, y_train, X_test, y_test, X_test2, y_test2, imagenet_to_objectnet, objectnet_to_imagenet)
    cluster_training_data(X_test2, y_test2,objectnet_to_imagenet)

# first scenario: train on imagenet, validate on ObjectNet, calculate accuracy only for samples from known classes
# first scenario: train on objectnet, validate on ObjectNet
# ResNet50
# k-means: 0.0370
# PCA+k-means: 0.223
# over (2): ARI 0.0936	V 0.6336	AMI 0.2520	FM 0.0969
#
# ACC TR L 0.2293	ACC TR M 0.2386	ACC VA L 0.2714	ACC VA M 0.2774

# PCA + ResNet-50
# ARI 0.1066	V 0.6182	AMI 0.3518	FM 0.1110
#
# ACC TR L 0.2325	ACC TR M 0.2546	ACC VA L 0.2534	ACC VA M 0.2769


# InfoMin
# ARI 0.1443	V 0.6867	AMI 0.4803	FM 0.1595
#
# ACC TR L 0.3301	ACC TR M 0.3665	ACC VA L 0.3473	ACC VA M 0.3830

# InfoMin large
# ARI 0.2122	V 0.7198	AMI 0.5249	FM 0.2224
#
# ACC TR L 0.3761	ACC TR M 0.4152	ACC VA L 0.3952	ACC VA M 0.4296


# over 4000
# ARI 0.1692	V 0.7380	AMI 0.4029	FM 0.1973
#
# ACC TR L 0.4611	ACC TR M 0.4630	ACC VA L 0.5092	ACC VA M 0.5120

# full pca + over 5
# ARI 0.1482	V 0.7468	AMI 0.3993	FM 0.1809
#
# ACC TR L 0.5110	ACC TR M 0.5117	ACC VA L 0.5550	ACC VA M 0.5566
