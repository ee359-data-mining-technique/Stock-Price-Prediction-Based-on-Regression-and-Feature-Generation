#!/bin/bash

mltype=MMC
distance=euclidean
neighbors=50


# paras for model

dims=2
k=50
gamma=0.5
bal=0.5
spa=0.5

python scripts/main.py --mltype $mltype --distance $distance --neighbors $neighbors\
                       --dims $dims --k $k --gamma $gamma --bal $bal --spa $spa
