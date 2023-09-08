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

class NLIDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, cut_rate):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cut_rate = cut_rate
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        this_data = self.data[index]
        sentence1 = this_data['src']
        sentence2 = this_data['tgt']
        label = this_data['label']
        encoded_p = self.tokenizer(sentence1, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        encoded_h = self.tokenizer(sentence2, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")

        a = math.ceil((torch.sum(encoded_p["attention_mask"].flatten(), dim=-1) * self.cut_rate).item())
        mes_label = torch.zeros(encoded_p["attention_mask"].flatten().shape)
        mes_label[0:a] = 1

        return dict(
                p=sentence1, 
                h=sentence2,
                input_ids_p=encoded_p["input_ids"].flatten(),
                attention_mask_p=encoded_p["attention_mask"].flatten(),
                input_ids_h=encoded_h["input_ids"].flatten(),
                attention_mask_h=encoded_h["attention_mask"].flatten(),
                label=torch.tensor(label),
                mse_label=mes_label
                )

class ColaDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, cut_rate):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cut_rate = cut_rate
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        this_data = self.data[index]
        sentence = this_data['src']
        label = this_data['label']
        encoded = self.tokenizer(sentence, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")

        a = math.ceil((torch.sum(encoded["attention_mask"].flatten(), dim=-1) * self.cut_rate).item())
        mes_label = torch.zeros(encoded["attention_mask"].flatten().shape)
        mes_label[0:a] = 1

        return dict(
                src=sentence,
                input_ids=encoded["input_ids"].flatten(),
                attention_mask=encoded["attention_mask"].flatten(),
                label=torch.tensor(label),
                mse_label=mes_label,
                )

class OneSentClassificationDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size, cut_rate, max_len=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.cut_rate = cut_rate
        self.predict_cola = False

        self.train_data = self.read_cola_data('dataset/CoLA/train.tsv')
        self.cola_dev_data = self.read_cola_data('dataset/CoLA/dev.tsv')

        self.conll_test_data = []
        with open(conll_test_src_path) as f:
            data = f.readlines()
            for d in data:
                this_data = {}
                this_data['src'] = d.strip()
                self.conll_test_data.append(this_data)
    
    def read_cola_data(self, data_path):
        import csv
        data = []
        with open(data_path) as f:
            tsv_file = csv.reader(f, delimiter="\t")
            for i, line in enumerate(tsv_file):
                if i == 0:
                    continue
                this_data = {}
                this_data['src'] = line[3]
                this_data['label'] = int(line[1])
                data.append(this_data)
        return data
    
    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train_dataset = ColaDataset(self.train_data, self.tokenizer, self.max_len, self.cut_rate)
        if stage == "predict" or stage is None:
            self.predict_conll_dataset = ChunkingDataset(self.conll_test_data, self.tokenizer, self.max_len)
            self.predict_cola_dataset = ChunkingDataset(self.cola_dev_data, self.tokenizer, self.max_len)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6)
    
    def predict_dataloader(self):
        if self.predict_cola:
            return DataLoader(self.predict_cola_dataset, batch_size=1)
        else:
            return DataLoader(self.predict_conll_dataset, batch_size=1)


class TwoSentClassificationDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size, cut_rate=0.5, max_len=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.cut_rate = cut_rate
        self.predict_snli = False

        # self.train_data = self.read_snli_data('dataset/snli_1.0/snli_1.0_train.jsonl')
        self.train_data = self.read_snli_data('dataset/multinli_1.0/multinli_1.0_train.jsonl')
        # self.snli_test_data = self.read_snli_data('dataset/snli_1.0/snli_1.0_test.jsonl', Test=True)
        self.snli_test_data = self.read_snli_data('dataset/multinli_1.0/multinli_1.0_dev_matched.jsonl', Test=True)

        self.conll_test_data = []
        with open(conll_test_src_path) as f:
            data = f.readlines()
            for d in data:
                this_data = {}
                this_data['src'] = d.strip()
                self.conll_test_data.append(this_data)
    
    def read_snli_data(self, data_path, Test=False):
        label2id = {"entailment":0, "contradiction":1, 'neutral':2}
        data = []
        sentences = []
        with open(data_path) as f:
            lines = f.readlines()
        i = 0
        for line in lines:
            this_data = {}
            tmp = json.loads(line)
            if not Test and tmp['gold_label'] not in list(label2id.keys()):
                continue
            this_data['id'] = i
            this_data['src'], this_data['tgt'] = tmp['sentence1'].strip(), tmp['sentence2'].strip()
            if Test and this_data['src'] in sentences:
                continue
            else:
                sentences.append(this_data['src'])
            if not Test:
                this_data['label'] = label2id[tmp['gold_label']]
            data.append(this_data)
            i += 1
        return data
    
    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train_dataset = NLIDataset(self.train_data, self.tokenizer, self.max_len, self.cut_rate)
        if stage == "predict" or stage is None:
            self.predict_conll_dataset = ChunkingDataset(self.conll_test_data, self.tokenizer, self.max_len)
            self.predict_snli_dataset = ChunkingDataset(self.snli_test_data, self.tokenizer, self.max_len)
    
    def train_dataloader(self):
        # torch.manual_seed(666)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6)
    
    def predict_dataloader(self):
        if self.predict_snli:
            return DataLoader(self.predict_snli_dataset, batch_size=1)
        else:
            return DataLoader(self.predict_conll_dataset, batch_size=1)

    
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    # d = DataModule(tokenizer, 8)
    # d.setup('test')
    # d = d.test_dataloader()
    # print(next(iter(d)))

    a = PretrainDataset(tokenizer, 128)
    for i in range(len(a)):
        a.getitem(i)

