import json

import matplotlib.pyplot as plt
import numpy as np
import requests
from nltk.corpus import wordnet
from scipy.cluster import hierarchy
from tqdm import trange

map_url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
response = json.loads(requests.get(map_url).text)

sims = np.zeros((2000, 2000))
cnt = 1000
ssets = []
labels = []
for i in trange(cnt):
    inst_i = str(i)
    ss_i = wordnet.synsets(response[inst_i][1])[0]
    ssets.append(ss_i)
    labels.append(response[inst_i][1])
for i in trange(cnt):
    ss_i = ssets[i]
    for j in range(cnt):
        if j == i:
            continue
        ss_j = ssets[j]
        sims[i][j] = wordnet.lch_similarity(ss_i, ss_j)

sz = np.zeros(2000, dtype=int)
sz[:1000] = np.ones(1000, dtype=int)
mask = np.zeros(2000, dtype=bool)
mask[:1000] = np.ones(1000, dtype=bool)
indices = np.arange(2000)
Z = np.zeros((999, 4))

for i in trange(999):
    idx1, idx2 = np.unravel_index(sims[mask][:, mask].argmax(), sims[mask][:, mask].shape)

    idx_full1, idx_full2 = (indices[mask])[idx1], (indices[mask])[idx2]
    assert idx_full1 != idx_full2
    Z[i][0] = idx_full1
    Z[i][1] = idx_full2
    Z[i][2] = sims[idx_full1, idx_full2]

    Z[i][3] = sz[idx_full1] + sz[idx_full2]
    sz[1000 + i] = sz[idx_full1] + sz[idx_full2]

    # update mask
    mask[idx_full1] = 0
    mask[idx_full2] = 0
    mask[1000 + i] = 1

    ss_new = ssets[idx_full1].lowest_common_hypernyms(ssets[idx_full2])[0]
    ssets.append(ss_new)
    # print("{}+{}={}".format(ssets[idx_full1], ssets[idx_full2], ss_new))
    for j in range(1000 + i):
        ss_j = ssets[j]
        sim = wordnet.lch_similarity(ss_new, ss_j)
        sims[1000 + i][j] = sim
        sims[j][1000 + i] = sim

# convert similarity to distance
mx = Z[:, 2].max() * 1.1
dist = 1 - Z[:, 2] / mx
for i in range(1, len(dist) - 1):
    if dist[i] < dist[i - 1]:
        if dist[i - 1] < dist[i + 1]:
            dist[i] = (dist[i - 1] + dist[i + 1]) / 2
        else:
            dist[i] = dist[i - 1] + 1e-5
assert (np.diff(dist) >= -1e-3).all()
Z[:, 2] = dist

fig = plt.figure(figsize=(66, 42))
# fig.tight_layout()
ax = plt.gca()
dn1 = hierarchy.dendrogram(Z, ax=ax, orientation='top', labels=labels)
plt.savefig('hier.pdf')
plt.show()
