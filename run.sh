#!/bin/bash

seed=1
dataset="adult"
datainpath="./LIBSVM/adult/a1a.t"
dataoutpath="./data"
outpath="./outputs"
noisesd=0.15
noiseds=0.15
datasize=15000
prior=0.7
numepochs=25
batchsize=128
runs=10
lname="sigmoid"
full=1


python3 data_generator.py \
    --seed=$seed \
    --dataset=$dataset \
    --datainpath=$datainpath \
    --dataoutpath=$dataoutpath \
    --noisesd=$noisesd \
    --noiseds=$noiseds \
    --datasize=$datasize \
    --prior=$prior

if [ $noisesd == $noiseds ]
then
    python3 un_train.py \
        --seed=$seed \
        --outpath=$outpath \
        --datapath=$dataoutpath \
        --numepochs=$numepochs \
        --batchsize=$batchsize \
        --runs=$runs \
        --lname=$lname \
        --noise=$noisesd
else
    python3 cc_train.py \
        --seed=$seed \
        --outpath=$outpath \
        --datapath=$dataoutpath \
        --numepochs=$numepochs \
        --batchsize=$batchsize \
        --runs=$runs \
        --lname=$lname \
        --full=$full
fi
