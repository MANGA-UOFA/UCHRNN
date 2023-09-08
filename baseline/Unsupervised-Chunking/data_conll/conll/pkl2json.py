import pickle, json

tokens = pickle.load(open("data_val_tokens.pkl",'rb'))
tags = pickle.load(open("data_val_tags.pkl",'rb'))

count = 0
data = {}
src_text_data = []
for d in zip(tokens,tags):
    this_data = []
    for token, tag in zip(d[0], d[1]):
        this_data.append([token, tag])
    data[str(count)] = this_data
    count += 1

    src_text_data.append(' '.join(d[0]))

with open('conll_devset_result.json', 'w') as f:
    json.dump(data, f, indent=4)

with open('conll_dev_src.txt', 'w') as f:
    f.write('\n'.join(src_text_data))
    
