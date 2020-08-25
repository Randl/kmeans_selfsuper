# Self-Supervised Learning for Large-Scale Unsupervised Image Clustering

This is code to run experiments for paper ["Self-Supervised Learning for Large-Scale Unsupervised Image Clustering"](https://arxiv.org/abs/2008.10312).

# Running the code
For part of the models, you'll need to download the chekpoints manually:
- SimCLR models: https://github.com/google-research/simclr
- MoCo and InfoMin models: https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/docs/MODEL_ZOO.md
- SwAV models: https://github.com/facebookresearch/swav

and put them in chekpoint folder.

For SimCLRv2, BigBiGAN as well as supervised models checkpoints are downloaded automatically.

## Download the code and install dependencies
Remember to clone the submodules by running 
```
git clone --recurse-submodules https://github.com/Randl/kmeans_selfsuper.git
```
during cloning the repo, or, if you forgot to do it, by running 
```
git submodule update --init --recursive
```
in the repo folder.

You'll need to install dependencies, by running
```
pip install -r requirements.txt
```
## Generating features
For SimCLRv2 and BigBiGAN run
```
python3 generate_prediction_tf.py --model resnet152_simclr2
python3 generate_prediction_tf.py --model resnet50_simclr2
python3 generate_prediction_tf.py --model resnet152x3_simclr2
python3 generate_prediction_tf.py --model resnet50_bigbigan
python3 generate_prediction_tf.py --model revnet50x4_bigbigan
```
For InfoMin, MoCo v2 and SwAV, run
```
python3 generate_prediction_pytorch.py --model resnext152_infomin
python3 generate_prediction_pytorch.py --model resnet50_infomin
python3 generate_prediction_pytorch.py --model resnet50_mocov2
python3 generate_prediction_pytorch.py --model resnet50_swav
```
Finally, for supervised models, run
```
python3 generate_prediction_pytorch_supervised.py --model tf_efficientnet_l2_ns_475
python3 generate_prediction_pytorch_supervised.py --model gluon_resnet152_v1s
python3 generate_prediction_pytorch_supervised.py --model ig_resnext101_32x48d
```
You'll need large amount of RAM since the script keeps features in memory. It was tested on machine with 128 GB RAM.
## Running clustering
To run clustering, you need to run
```
python3 cluster.py --model resnet50_infomin
```
where the model name should fit the name in generating part. For overclustering, e.g., 1.25 times more clusters 
than classes, run
```
python3 cluster.py --model resnet152_simclr2 --over 1.25
```
For using smaller dimensions of features, e.g., 512, run
```
python3 cluster.py --model resnet152_simclr2 --n-components 512
```
# Citing the paper
If you found the paper or the code useful, please cite it. You can use following bibtex entry:
```
@article{zheltonozhskii2020unsupevised,
  title={Self-Supervised Learning for Large-Scale Unsupervised Image Clustering},
  author={Zheltonozhskii, Evgenii and Baskin, Chaim and Bronstein, Alex M. and Mendelson, Avi},
  journal={arXiv preprint arXiv:2008.10312},
  year={2020},
  url = {https://arxiv.org/abs/2008.10312}
}
```