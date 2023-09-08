import random
import json

def create_dict(src_path, tgt_path, replace_unk=True):
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

giga_path = '../dataset/ggw_data/org_data/'
test_src_path = giga_path + 'test.src.txt'
test_tgt_path = giga_path + 'test.tgt.txt'

test_data = create_dict(test_src_path, test_tgt_path)
avg_chunk = 16 - 1

random.seed(666)
out_dict = {}
for data_index, d in enumerate(test_data):
    src = d['src']
    src_words = src.split()

    new_src_words = ['|']
    if len(src_words) >= avg_chunk:
        src_indice = list(range(len(src_words)))
        chosen_indice = random.sample(src_indice, avg_chunk) 
        for i, word in enumerate(src_words):
            new_src_words.append(word)
            if i in chosen_indice:
                new_src_words.append('|')
    else:
        for word in src_words:
            new_src_words.append(word)
            new_src_words.append('|')
    
    new_src = ' '.join(new_src_words)
    out_dict[data_index] = new_src

with open("random_chunker.json", "w") as of:
    json.dump(out_dict, of, indent=4)





    
    

    
    

