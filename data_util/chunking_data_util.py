from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
import math
import numpy as np
import json

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

# class ChunkingDataset(Dataset):
#     def __init__(self, data, tokenizer, max_len, return_offset=True):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.return_offset = return_offset
    
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index: int):
#         this_data = self.data[index]
#         source = this_data['src']
#         encoded_src = self.tokenizer(source, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt", return_offsets_mapping=True)
#         concat_indice = my_decode(encoded_src)
#         assert concat_indice[-1][-1] + 2 == torch.sum(encoded_src['attention_mask']).item()
#         # assert len(source.split()) == len(concat_indice)

#         return dict(
#                 text=source, 
#                 input_ids=encoded_src["input_ids"].flatten(),
#                 attention_mask=encoded_src["attention_mask"].flatten(),
#                 offset_mapping=concat_indice
#                 )

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