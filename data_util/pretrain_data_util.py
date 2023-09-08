from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
import math
import numpy as np
import json
from .chunking_data_util import ChunkingDataset

conll_train_src_path = "./dataset/conll_2000/conll_pretrain_src.txt" 
conll_train_tgt_path = "./dataset/conll_2000/conll_pretrain_tgt.txt" # word level
conll_test_src_path = "./dataset/conll_2000/test_conll_src.txt"
conll_test_tgt_path = "./dataset/conll_2000/test_conll_tgt.txt"

class PretrainDataset(Dataset):
    def __init__(self, src_path, tgt_path, tokenizer, max_len):
        self.src_path = src_path 
        self.tgt_path = tgt_path
        self.data = self.handle_raw_data()
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def handle_raw_data(self):
        with open(self.src_path) as sf:
            src_data = sf.readlines()
        with open(self.tgt_path) as tf:
            tgt_data = tf.readlines()

        assert len(src_data) == len(tgt_data)
        # max_len = 0
        data = []
        for i, line in enumerate(zip(src_data, tgt_data)):
            this_dict = {}
            this_src = line[0].strip()
            this_tgt = [int(x) for x in line[1].strip().split() if x != '2']
            assert len(this_src.split()) == len(this_tgt)
            # if len(this_tgt) > max_len:
            #     max_len = len(this_tgt)
            this_dict['src'] = this_src
            this_dict['tgt'] = this_tgt
            data.append(this_dict)
        # print(len(data))
        return data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        this_data = self.data[index]
        source = this_data['src']
        target = this_data['tgt']

        encoded_src = self.tokenizer(source, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt", return_offsets_mapping=True)
        offset_mapping = encoded_src['offset_mapping']

        concat_indice = []
        last_index = 0
        for word in source.split():
            this_indice = []
            this_text = ''
            for i, offset in enumerate(offset_mapping.tolist()[0]):
                if i <= last_index:
                    continue
                this_text += source[offset[0]:offset[1]]
                # this_text += source[offset[0]:offset[1]].strip() # some tokenizer preseve the space which may cause error
                this_indice.append(i)
                if this_text == word:
                    last_index = i
                    break
            concat_indice.append(this_indice)
        assert len(concat_indice) == len(target)

        token_level_target = [0] # pad start token
        for item in zip(concat_indice, target):
            label = [0] * len(item[0])
            label[-1] = item[1]
            token_level_target += label
        token_level_target += [0] # pad end token
        
        assert len(token_level_target) == torch.sum(encoded_src['attention_mask']).item()

        # start end token is not included in training
        attention_mask = torch.roll(encoded_src['attention_mask'], -1)
        attention_mask[0][0] = 0
        attention_mask[0][-1] = 0
        
        token_level_target = torch.tensor(token_level_target).unsqueeze(0)
        token_level_target = torch.nn.functional.pad(token_level_target, pad=(0,self.max_len-token_level_target.shape[1],0,0))
        
        return dict(
                src_text=source, 
                tgt=token_level_target.flatten(), 
                src_input_ids=encoded_src["input_ids"].flatten(),
                src_attention_mask=attention_mask.flatten(),
                )
    

class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size, max_src_len=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.train_dataset = PretrainDataset(conll_train_src_path, conll_train_tgt_path, self.tokenizer, self.max_src_len)

        with open(conll_test_src_path) as f:
            data = f.readlines()
            src_data = []
            for d in data:
                this_data = {}
                this_data['src'] = d.strip()
                src_data.append(this_data)

        self.test_dataset = ChunkingDataset(src_data, self.tokenizer, self.max_src_len)
    
    def setup(self, stage):
        pass
    
    def __len__(self):
        return len(self.train_dataset)

    def train_dataloader(self):
        # torch.manual_seed(20200503)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)