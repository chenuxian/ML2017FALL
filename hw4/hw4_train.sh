#!/bin/bash
printf "*************************************************************************************************************************************************\n"
printf "semi-supervised training i will use testing data to train together, so please call bash hw4_train.sh 'train data' 'train nolable data' 'test data'\n"
printf "*************************************************************************************************************************************************\n"
python3 hw4.py sup train --train_data $1
python3 hw4.py semi semi --load_model './' --train_data $1 --semi_data $2 --test_data $3
