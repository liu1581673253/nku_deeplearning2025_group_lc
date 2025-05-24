#! /bin/bash

python joint_main.py --arch resnet --train_root ./data/DUTS/DUTS-TR --train_list ./data/DUTS/DUTS-TR/train_pair.lst --train_edge_root ./data/DUTS/DUTS-TR --train_edge_list ./data/DUTS/DUTS-TR/edges.lst
# you can optionly change the -lr and -wd params
