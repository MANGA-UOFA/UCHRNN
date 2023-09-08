import nltk
import json
from nltk.corpus import conll2000

#grammar = "NP: {<DT>?<JJ>*<NN>}"
#grammar = r"NP: {<[CDJNP].*>+}"
 	
# grammar = r"""
#   NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
#   PP: {<IN><NP>}               # Chunk prepositions followed by NP
#   VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
#   CLAUSE: {<NP><VP>}           # Chunk NP, VP
#   """


# chunker = nltk.RegexpParser(grammar)

def read_snli_data(data_path):
    data = []
    with open(data_path) as f:
        lines = f.readlines()
    for line in lines:
        tmp = json.loads(line)
        if tmp['gold_label'] != 'entailment':
            continue
        data.append(tmp['sentence1'])
    return data


class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

# nltk.download('conll2000')
# nltk.download('averaged_perceptron_tagger')
train_sents = conll2000.chunked_sents('train.txt')

chunker = BigramChunker(train_sents)

# file = open("../dataset/ggw_data/org_data/selected_dev.src.txt", "r")
# text = file.read().split("\n")[:-1]

# text = conll2000.sents('test.txt')
sentence_chunks = dict()
# out_file = open("../result_files/gt/giga_devset_result.json", "w")
out_file = open("mnli_mismatched_result.json", "w")

text = read_snli_data('../dataset/multinli_1.0/multinli_1.0_dev_mismatched.jsonl')
print(len(text))

# def extractChunk(tree, sentence_id):
#     chunks = []
#     for c in range(len(tree)):
#         chunk = tree[c]
#         if isinstance(chunk, tuple):
#             if chunk[0] not in [",", ".", ";", ":", "``", "--", "-rrb-", "-lrb-", "...", "-", "'", "`", "?", "!"]:
#                 if chunk[0] == "UNK":
#                     chunks.append(("<unk>", "B"))
#                 else:
#                     chunks.append((chunk[0], "B"))
#             continue
#         first_word = chunk[0][0]
#         if first_word == "UNK":
#             chunks.append(("<unk>", "B"))
#         else:
#             chunks.append((first_word, "B"))
        
#         for i in range(len(chunk)):
#             if i == 0:
#                 continue
#             subtree = chunk[i]
#             if isinstance(subtree, tuple):
#                 if subtree[0] == "UNK":
#                     chunks.append(("<unk>", "I"))
#                 else:                    
#                     chunks.append((subtree[0], "I"))
#             elif isinstance(subtree, nltk.tree.Tree):
#                 for j in range(len(subtree)):
#                     word = subtree[j][0]
#                     if word == "UNK":
#                         chunks.append(("<unk>", "I"))
#                     else:
#                         chunks.append((word, "I"))
#     sentence_chunks[sentence_id] = chunks
    
def extractChunk(tree, sentence_id):
    chunks = []
    for c in range(len(tree)):
        chunk = tree[c]
        if isinstance(chunk, tuple):
            if chunk[0] not in [",", ".", ";", ":", "``", "--", "-rrb-", "-lrb-", "...", "-", "'", "`", "?", "!", "-----"]:
                if chunk[0] == "UNK":
                    chunks.append(("<unk>", "B"))
                else:
                    chunks.append((chunk[0], "B"))
            continue
        first_word = True
        
        for i in range(len(chunk)):
            subtree = chunk[i]
            if isinstance(subtree, tuple):
                if subtree[0] in [",", ".", ";", ":", "``", "--", "-rrb-", "-lrb-", "...", "-", "'", "`", "?", "!", "-----"]:
                    continue
                if subtree[0] == "UNK":
                    if first_word:
                        chunks.append(("<unk>", "B"))
                        first_word = False
                    else:
                        chunks.append(("<unk>", "I"))
                else:          
                    if first_word:          
                        chunks.append((subtree[0], "B"))
                        first_word = False
                    else:
                        chunks.append((subtree[0], "I"))
            elif isinstance(subtree, nltk.tree.Tree):
                for j in range(len(subtree)):
                    word = subtree[j][0]
                    if word in [",", ".", ";", ":", "``", "--", "-rrb-", "-lrb-", "...", "-", "'", "`", "?", "!", "-----"]:
                        continue
                    if word == "UNK":
                        if first_word:
                            chunks.append(("<unk>", "B"))
                            first_word = False
                        else:
                            chunks.append(("<unk>", "I"))
                    else:
                        if first_word:
                            chunks.append((word, "B"))
                            first_word = False
                        else:
                            chunks.append((word, "I"))
    sentence_chunks[sentence_id] = chunks  
    


for i in range(len(text)):
    tokens = nltk.word_tokenize(text[i])
    # tokens = text[i].split()
    # tokens = text[i]
    tokens_tag = nltk.pos_tag(tokens)
    #print("After Token:",tokens_tag)
    chunks = chunker.parse(tokens_tag)
    extractChunk(chunks, i)

json.dump(sentence_chunks, out_file, indent=6)





