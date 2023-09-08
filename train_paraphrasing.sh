#!/bin/bash

fname_test='./result_files/train_mnli_predict_mnli.json'
fname_conll='./result_files/train_mnli_predict_mnli_conll.json'

target_file_test='./result_files/gt/mnli_matched_result.json'
target_file_conll='./result_files/gt/conll_testset_result.json'

CUDA_VISIBLE_DEVICES=2 python main_train_generation.py \
--load_checkpoint \
--layers_ratio 1 \
--heads_ratio 1 \
--cut_rate 0.6 \
--learning_rate 4e-5 \
--reweight_coefficient 0.1 \
--limit_train_batches 800 \
--fname_test $fname_test \
--fname_conll $fname_conll \
--train_dataset mnli \
--test_dataset mnli \
--predict_dataset wmt

python evaluate.py \
--target_file $target_file_test \
--fname $fname_test

python evaluate.py \
--target_file $target_file_conll \
--fname $fname_conll

