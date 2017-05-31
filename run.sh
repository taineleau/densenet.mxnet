#!/usr/bin/env sh
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
#export MXNET_ENGINE_TYPE=NaiveEngine
#export MXNET_BACKWARD_DO_MIRROR=1

python2 -u train_densenet.py --data-dir ./cifar --data-type cifar10 --depth 100 --batch-size 200 --growth-rate 12 --drop-out 0 --reduction 0.5 --gpus=0 --aug-level  1
