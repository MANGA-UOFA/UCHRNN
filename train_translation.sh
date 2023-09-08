#!/bin/bash

fname_indomain='./result_files/train_wmt_predict_wmt.json'
fname_conll='./result_files/train_wmt_predict_wmt_conll.json'

target_file_indomain='./translation/result_files/gt/wmt_en_result.json'
target_file_conll='./result_files/gt/conll_testset_result.json'


cd translation

CUDA_VISIBLE_DEVICES=3 python main_train_translation.py \
--load_checkpoint \
--limit_train_batches 4000 \
--fname_indomain $fname_indomain \
--fname_conll $fname_conll \
--predict_dataset giga \
--layers_ratio 1 \
--heads_ratio 1 \
--reweight_coefficient 0.1

cd ..

python evaluate.py \
--target_file $target_file_indomain \
--fname translation/$fname_indomain

python evaluate.py \
--target_file $target_file_conll \
--fname translation/$fname_conll




