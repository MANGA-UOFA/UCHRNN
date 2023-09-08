from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
import math
import numpy as np
import json
import nltk

conll_train_src_path = "../dataset/conll_2000/conll_pretrain_src.txt" 
conll_train_tgt_path = "../dataset/conll_2000/conll_pretrain_tgt.txt" # word level
conll_test_src_path = "../dataset/conll_2000/test_conll_src.txt"
conll_test_tgt_path = "../dataset/conll_2000/test_conll_tgt.txt"
conll_dev_src_path = "../dataset/conll_2000/dev_conll_src.txt"

giga_path = '../dataset/ggw_data/org_data/'
giga_train_src_path = giga_path + 'train.src.txt'
giga_train_tgt_path = giga_path + 'train.tgt.txt'
giga_dev_src_path = giga_path + 'selected_dev.src.txt'
giga_dev_tgt_path = giga_path + 'selected_dev.tgt.txt'
giga_test_src_path = giga_path + 'test.src.txt'
giga_test_tgt_path = giga_path + 'test.tgt.txt'

mnli_path = '../dataset/multinli_1.0/'
mnli_train_path = mnli_path + 'multinli_1.0_train.jsonl'
mnli_dev_path = mnli_path + 'multinli_1.0_dev_mismatched.jsonl'
mnli_test_path = mnli_path + 'multinli_1.0_dev_matched.jsonl'

def my_decode(encoded):
    # https://stackoverflow.com/questions/62317723/tokens-to-words-mapping-in-the-tokenizer-decode-step-huggingface
    desired_output = []
    for word_id in encoded.word_ids():
        if word_id is not None:
            start, end = encoded.word_to_tokens(word_id)
            if start == end - 1:
                tokens = [start]
            else:
                tokens = []
                for i in range(start, end):
                    tokens.append(i)
            if len(desired_output) == 0 or desired_output[-1] != tokens:
                desired_output.append(tokens)
    return desired_output

def create_dict(src_path, tgt_path):
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

class ChunkingDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, return_offset=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.return_offset = return_offset
        self.data = self.handle_raw_data(data)

    def handle_raw_data(self, data):
        filtered_data = []
        count = 0
        for this_data in data:
            source = this_data['src']
            encoded_src = self.tokenizer(source, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt", return_offsets_mapping=True)
            concat_indice = my_decode(encoded_src)
            try:
                assert concat_indice[-1][-1] + 2 == torch.sum(encoded_src['attention_mask']).item()
                filtered_data.append((source, encoded_src["input_ids"], encoded_src["attention_mask"], concat_indice))
            except AssertionError:
                count += 1
        print('%d samples filtered...' % count)
        return filtered_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        this_data = self.data[index]

        return dict(
                text=this_data[0],
                input_ids=this_data[1].flatten(),
                attention_mask=this_data[2].flatten(),
                offset_mapping=this_data[3],
                )


class PretrainDataset(Dataset):
    def __init__(self, src_path, tgt_path, tokenizer, max_len):
        self.src_path = src_path 
        self.tgt_path = tgt_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self.handle_raw_data()
    
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

        count = 0
        filtered_data = []
        for this_data in data:
            source = this_data['src']
            target = this_data['tgt']
            result = self.process_data(source, target)
            if result != None:
                filtered_data.append(result)
            else:
                count += 1
        print('%d samples filtered...' % count)
        return filtered_data

    def process_data(self, source, target):
        encoded_src = self.tokenizer(source, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt", return_offsets_mapping=True)
        offset_mapping = encoded_src['offset_mapping']

        concat_indice = []
        last_index = 0
        for word in source.split():
            # print(word)
            this_indice = []
            this_text = ''
            for i, offset in enumerate(offset_mapping.tolist()[0]):
                if i <= last_index:
                    continue
                this_text += source[offset[0]:offset[1]].strip()
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
        
        try:
            assert len(token_level_target) == torch.sum(encoded_src['attention_mask']).item()
        except AssertionError:
            return None

        # start end token is not included in training
        attention_mask = torch.roll(encoded_src['attention_mask'], -1)
        attention_mask[0][0] = 0
        attention_mask[0][-1] = 0
        
        token_level_target = torch.tensor(token_level_target).unsqueeze(0)
        token_level_target = torch.nn.functional.pad(token_level_target, pad=(0,self.max_len-token_level_target.shape[1],0,0))

        return (source, token_level_target, encoded_src["input_ids"], attention_mask)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        this_data = self.data[index]
        
        return dict(
                src_text=this_data[0], 
                tgt=this_data[1].flatten(), 
                src_input_ids=this_data[2].flatten(),
                src_attention_mask=this_data[3].flatten(),
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
        torch.manual_seed(666)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)


class TranslationDataset(Dataset):
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
        with self.tokenizer.as_target_tokenizer():
            encoded_tgt = self.tokenizer(target, max_length=self.max_tgt_len, padding="max_length", truncation=True, return_tensors="pt")
        # encoded_tgt = self.tokenizer(text_target=target, max_length=self.max_tgt_len, padding="max_length", truncation=True, return_tensors="pt")

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

class TranslationDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, hparams, max_src_len=128, max_tgt_len=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = hparams.batch_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.predict_wmt = True
        self.cut_rate = hparams.cut_rate
        self.predict_dataset = hparams.predict_dataset
        self.train_data = create_dict('./WMT14-en-de/train.en', './WMT14-en-de/train.de')
        self.test_data = create_dict('./WMT14-en-de/newstest2014.en', './WMT14-en-de/newstest2014.de')
        
        if self.predict_dataset == 'wmt':
            self.predict_data = self.test_data
            print('predict wmt')
        elif self.predict_dataset == 'giga':
            self.predict_data = create_giga_dict(giga_test_src_path, giga_test_tgt_path)
            print('predict giga')
        elif self.predict_dataset == 'mnli':
            self.predict_data= read_snli_entailment_data(mnli_test_path)
            print('predict mnli')
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
    
    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train_dataset = TranslationDataset(self.train_data, self.tokenizer, self.max_src_len, self.max_tgt_len, self.cut_rate)
        if stage == "test" or stage is None:
            self.test_dataset = TranslationDataset(self.test_data, self.tokenizer, self.max_src_len, self.max_tgt_len, self.cut_rate)
        if stage == "predict" or stage is None:
            self.predict_ind_dataset = ChunkingDataset(self.predict_data, self.tokenizer, self.max_src_len)
            self.predict_conll_dataset = ChunkingDataset(self.conll_test_data, self.tokenizer, self.max_src_len)

    def train_dataloader(self):
        torch.manual_seed(59)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        if self.predict_wmt:
            return DataLoader(self.predict_ind_dataset, batch_size=1)
        else:
            return DataLoader(self.predict_conll_dataset, batch_size=1)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from transformers import MBartForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="de_DE")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    data = create_dict('./WMT14-en-de/train.en', './WMT14-en-de/train.de')
    for d in data:
        encoded_src = tokenizer(d['src'], return_tensors="pt")
        encoded_tgt = tokenizer(text_target=d['tgt'], return_tensors="pt")
        # print(encoded_src, encoded_tgt)
        # print(tokenizer.batch_decode(encoded_src['input_ids'], skip_special_tokens=True))
        print(tokenizer.batch_decode(encoded_tgt['input_ids'], skip_special_tokens=True))
        out = model.generate(**encoded_src, forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"])
        decoded_out = tokenizer.batch_decode(out, skip_special_tokens=True)
        print(decoded_out)
        exit()