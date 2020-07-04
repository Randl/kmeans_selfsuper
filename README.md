SimCLR models: https://github.com/google-research/simclr
BigBiGAN models: automatically from tf.hub
MoCo and InfoMin models: https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/docs/MODEL_ZOO.md

SimCLRv2: 
Install google-cloud-sdk and
gsutil cp -r gs://simclr-checkpoints/simclrv2/pretrained/r152_3x_sk1 .

python3 generate_prediction_pytorch.py --model resnext152_infomin
python3 generate_prediction_tf.py --model resnet152_simclr2
python3 generate_prediction_tf.py --model resnet50_simclr2
python3 generate_prediction_pytorch.py --model resnet50_infomin
python3 generate_prediction_tf.py --model resnet152x3_simclr2
python3 generate_prediction_tf.py --model resnet50_bigbigan
python3 generate_prediction_tf.py --model revnet50x4_bigbigan
python3 generate_prediction_pytorch.py --model resnet50_mocov2