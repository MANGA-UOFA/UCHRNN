from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
import math
import numpy as np
import json
from .chunking_data_util import ChunkingDataset
import nltk

giga_path = './dataset/ggw_data/org_data/'
giga_train_src_path = giga_path + 'train.src.txt'
giga_train_tgt_path = giga_path + 'train.tgt.txt'
giga_dev_src_path = giga_path + 'selected_dev.src.txt'
giga_dev_tgt_path = giga_path + 'selected_dev.tgt.txt'
giga_test_src_path = giga_path + 'test.src.txt'
giga_test_tgt_path = giga_path + 'test.tgt.txt'

conll_train_src_path = "./dataset/conll_2000/conll_pretrain_src.txt" 
conll_train_tgt_path = "./dataset/conll_2000/conll_pretrain_tgt.txt" # word level
conll_test_src_path = "./dataset/conll_2000/test_conll_src.txt"
conll_test_tgt_path = "./dataset/conll_2000/test_conll_tgt.txt"
conll_dev_src_path = "./dataset/conll_2000/dev_conll_src.txt"

snli_path = './dataset/snli_1.0/'
snli_train_path = snli_path + 'snli_1.0_train.jsonl'
snli_dev_path = snli_path + 'snli_1.0_dev.jsonl'
snli_test_path = snli_path + 'snli_1.0_test.jsonl'

mnli_path = './dataset/multinli_1.0/'
mnli_train_path = mnli_path + 'multinli_1.0_train.jsonl'
mnli_dev_path = mnli_path + 'multinli_1.0_dev_mismatched.jsonl'
mnli_test_path = mnli_path + 'multinli_1.0_dev_matched.jsonl'

wmt_train_src_path = './translation/WMT14-en-de/train.en'
wmt_train_tgt_path =  './translation/WMT14-en-de/train.de'
wmt_test_src_path = './translation/WMT14-en-de/newstest2014.en'
wmt_test_tgt_path = './translation/WMT14-en-de/newstest2014.de'

def create_wmt_dict(src_path, tgt_path):
    data = []
    with open(src_path) as sf:
        with open(tgt_path) as tf:
            for id, line in enumerate(zip(sf, tf)):
                dict = {}
                src, tgt = line[0], line[1]
                dict['src'] = src.strip().replace(" ##AT##-##AT## ", "-")
                dict['tgt'] = tgt.strip().replace(" ##AT##-##AT## ", "-")
                data.append(dict)
    return data

def create_giga_dict(src_path, tgt_path, replace_unk=True):
    data = []
    with open(src_path) as sf:
        with open(tgt_path) as tf:
            for id, line in enumerate(zip(sf, tf)):
                dict = {}
                src, tgt = line[0], line[1]
                if replace_unk:
                    src = src.strip().replace("UNK", "<unk>")
                    tgt = tgt.strip().replace("UNK", "<unk>")
                else:
                    src = src.strip()
                    tgt = tgt.strip()
                dict['src'] = src
                dict['tgt'] = tgt
                data.append(dict)
    return data

def read_snli_entailment_data(data_path):
        data = []
        with open(data_path) as f:
            lines = f.readlines()
        for line in lines:
            this_data = {}
            tmp = json.loads(line)
            if tmp['gold_label'] == 'entailment':
                this_data['src'] = ' '.join(nltk.word_tokenize(tmp['sentence1'].strip()))
                this_data['tgt'] = ' '.join(nltk.word_tokenize(tmp['sentence2'].strip()))
                data.append(this_data)
        return data

class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_src_len, max_tgt_len, cut_rate):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.cut_rate = cut_rate
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        this_data = self.data[index]
        source = this_data['src']
        target = this_data['tgt']
        encoded_src = self.tokenizer(source, max_length=self.max_src_len, padding="max_length", truncation=True, return_tensors="pt")
        encoded_tgt = self.tokenizer(target, max_length=self.max_tgt_len, padding="max_length", truncation=True, return_tensors="pt")

        # cut_rate = np.clip(np.random.normal(0.6, 0.1), 0, 1)
        a = math.ceil((torch.sum(encoded_src["attention_mask"].flatten(), dim=-1) * self.cut_rate).item())
        mes_label = torch.zeros(encoded_src["attention_mask"].flatten().shape)
        mes_label[0:a] = 1

        return dict(
                src_text=source, 
                tgt_text=target, 
                src_input_ids=encoded_src["input_ids"].flatten(),
                src_attention_mask=encoded_src["attention_mask"].flatten(),
                tgt_input_ids=encoded_tgt["input_ids"].flatten(),
                tgt_attention_mask=encoded_tgt["attention_mask"].flatten(),
                mse_label=mes_label,
                )

class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, hparams, max_src_len=128, max_tgt_len=64):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = hparams.batch_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.train_dataset = hparams.train_dataset
        self.test_dataset = hparams.test_dataset
        self.predict_dataset = hparams.predict_dataset
        self.cut_rate = hparams.cut_rate
        self.predict_mode = None

        if self.train_dataset == 'giga':
            self.train_data = create_giga_dict(giga_train_src_path, giga_train_tgt_path)
        elif self.train_dataset == 'mnli':
            self.train_data = read_snli_entailment_data(mnli_train_path)
        else:
            exit('No such train dataset')
        
        if self.test_dataset == 'giga':
            self.test_data = create_giga_dict(giga_test_src_path, giga_test_tgt_path)
        elif self.test_dataset == 'mnli':
             self.test_data = read_snli_entailment_data(mnli_test_path)
        else:
            exit('No such test dataset')
        
        if self.predict_dataset == 'wmt' :
            self.predcit_data = create_wmt_dict(wmt_test_src_path, wmt_test_tgt_path) 
        elif self.predict_dataset == 'giga' :
            self.predcit_data = create_giga_dict(giga_test_src_path, giga_test_tgt_path)
        elif self.predict_dataset == 'mnli':
            self.predcit_data= read_snli_entailment_data(mnli_test_path)
        else:
            exit('No such predict dataset')
        
        self.conll_test_data = []
        with open(conll_test_src_path) as f:
        # with open(conll_dev_src_path) as f:
            data = f.readlines()
            for d in data:
                this_data = {}
                this_data['src'] = d.strip()
                self.conll_test_data.append(this_data)
        print('Done!')
    
    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train_dataset = SummarizationDataset(self.train_data, self.tokenizer, self.max_src_len, self.max_tgt_len, self.cut_rate)
            # self.dev_dataset = SummarizationDataset(self.dev_data, self.tokenizer, self.max_src_len, self.max_tgt_len, self.cut_rate)
        if stage == "test" or stage is None:
            self.test_dataset = SummarizationDataset(self.test_data, self.tokenizer, self.max_src_len, self.max_tgt_len, self.cut_rate)
        if stage == "predict" or stage is None:
            self.predict_dataset = ChunkingDataset(self.predcit_data, self.tokenizer, self.max_src_len)
            self.predict_conll_dataset = ChunkingDataset(self.conll_test_data, self.tokenizer, self.max_src_len)

    def train_dataloader(self):
        # torch.manual_seed(42)
        torch.manual_seed(59)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)
    
    # def val_dataloader(self):
    #     return DataLoader(self.dev_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        if self.predict_mode == 'test':
            return DataLoader(self.predict_dataset, batch_size=1)
        elif self.predict_mode == 'conll':
            return DataLoader(self.predict_conll_dataset, batch_size=1)
        # elif self.predict_mode == 'dev':
        #     return DataLoader(self.predict_val_dataset, batch_size=1)
        else:
            print('predict: No such option!')
