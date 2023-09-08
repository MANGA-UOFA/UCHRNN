#!/bin/bash

fname_test='./result_files/train_giga_predict_giga.json'
fname_conll='./result_files/train_giga_predict_giga_conll.json'

target_file_test='./result_files/gt/giga_testset_result.json'
target_file_conll='./result_files/gt/conll_testset_result.json'

CUDA_VISIBLE_DEVICES=1 python main_train_generation.py \
--load_checkpoint \
--layers_ratio 1 \
--heads_ratio 1 \
--cut_rate 0.5 \
--learning_rate 5e-5 \
--reweight_coefficient 0.1 \
--limit_train_batches 1000 \
--fname_test $fname_test \
--fname_conll $fname_conll \
--train_dataset giga \
--test_dataset giga \
--predict_dataset wmt

python evaluate.py \
--target_file $target_file_test \
--fname $fname_test

python evaluate.py \
--target_file $target_file_conll \
--fname $fname_conll

