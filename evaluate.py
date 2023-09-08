import json, os, string
from argparse import ArgumentParser
from transformers import AutoTokenizer
import numpy as np
import subprocess
import statistics

class Evaluation:
    def __init__(self, fname, target_file) -> None:
        self.source_file = fname
        self.middle_file = fname[:-5]+'_BI'+fname[-5:]
        self.compare_file = fname[:-5]+'_final.txt'
        self.target_file = target_file

    def write_BI_form(self, source_file, middle_file):
        # file = open("./result_files/model_testset_result_cutoff="+cutoff+".json", "r")
        # store_file = open("./result_files/model_test_BI_cutoff="+cutoff+".json",  "w")
        
        file = open(source_file, "r")
        store_file = open(middle_file,  "w")

        data = json.load(file)
        result = {}
        for i in data.keys():
            text = data[i]
            chunks = text.split("|")
            word_list = []
            for j in range(len(chunks)):
                if chunks[j] == "":
                    continue
                chunk = chunks[j].strip().split()
                first_word = True
                for word in chunk:
                    if word in [",", ".", ";", ":", "``", "--", "-rrb-", "-lrb-", "...", "-", "'", "`", "?", "!", "-----"]:
                        continue
                    if first_word:
                        word_list.append([word, "B"])
                        first_word = False
                    else:
                        word_list.append([word, "I"])
        
            result[i] = word_list
        
        json.dump(result, store_file, indent=6)
        
    def compare_result(self, middle_file, target_file, compare_file):
        nltk_file = open(target_file, "r")
        # hrnn_file = open("./result_files/model_test_BI_cutoff="+cutoff+".json",  "r")
        # compare_file = open("evaluate_cutoff="+cutoff+".txt", "w")
        
        hrnn_file = open(middle_file,  "r")
        compare_file = open(compare_file, "w")
        
        nltk = json.load(nltk_file)
        hrnn = json.load(hrnn_file)
        
        for key in nltk.keys():
            try:
                nltk_result = nltk[key]
                hrnn_result = hrnn[key]
                len1 = len(nltk_result)
                len2 = len(hrnn_result)
                for i in range(len1):
                    nltk_tag = nltk_result[i][1]
                    hrnn_tag = hrnn_result[i][1]
                    nltk_word = nltk_result[i][0]
                    hrnn_word = hrnn_result[i][0]
                    if nltk_word != hrnn_word:
                        # print(nltk_word, hrnn_word)
                        continue
                    compare_file.write("x y " + nltk_tag + " " + hrnn_tag + "\n")
            except:
                pass
        compare_file.close()       
        
        # print("done")
          
    # def count(self, fname):
    #     nltk_file = open(fname, "r")
    #     nltk = json.load(nltk_file)
        
    #     tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

    #     overall_ratio = []
    #     for key in nltk.keys():
    #         this_B, this_I, len1 = 0, 0, 0
    #         nltk_result = nltk[key]
    #         for item in nltk_result:
    #             nltk_word = item[0]
    #             len_token = np.sum(tokenizer(nltk_word)['attention_mask'], axis=-1) - 2
    #             nltk_tag = item[1]
    #             if nltk_tag == "B":
    #                 this_B += 1  
    #             else:
    #                 this_I += len_token
    #             len1 += len_token
    #         this_ratio = (this_B + 2)/len1
    #         if this_ratio != 1:
    #             overall_ratio.append(this_ratio)

    #     return statistics.mean(overall_ratio), statistics.stdev(overall_ratio)

    def run_perl_script_with_file_input(self, script_path, file_path):
        with open(file_path, 'r') as file:
            result = subprocess.run(["perl", script_path], stdin=file, capture_output=True, text=True)
        return result.stdout

    def do_evaluate(self):
        self.write_BI_form(self.source_file, self.middle_file)
        # real_mean, real_std = self.count(self.middle_file)
        self.compare_result(self.middle_file, self.target_file, self.compare_file)


        result = self.run_perl_script_with_file_input('eval_conll2000_updated.pl', self.compare_file)
        tag_acc, phrase_f1 = result.split()
        return float(phrase_f1), float(tag_acc)
        # os.system("perl eval_conll2000_updated.pl < %s" % compare_file)

    
if __name__ == "__main__":
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = ArgumentParser()
    parser.add_argument("--fname", default="./result_files/model_testset_result.json", type=str)
    parser.add_argument("--target_file", default="./result_files/gt/conll_testset_result.json", type=str)
    args = parser.parse_args()

    e = Evaluation(args.fname, args.target_file)
    
    f1, acc = e.do_evaluate()
    print('Phrase F1: %.2f; Tag acc: %.2f' % (f1, acc))

    # if args.dataset == 'giga':
    #     do_evaluate(args.fname, target_file="./result_files/gt/giga_testset_result.json")
    # elif args.dataset == 'conll':
    #     do_evaluate(args.fname, target_file="./result_files/gt/conll_testset_result.json")
    # elif args.dataset == 'snli':
    #     do_evaluate(args.fname, target_file="./result_files/gt/snli_testset_result.json")
    # elif args.dataset == 'mnli':
    #     do_evaluate(args.fname, target_file="result_files/gt/mnli_matched_result.json")
    # elif args.dataset == 'cola':
    #     do_evaluate(args.fname, target_file="result_files/gt/cola_dev_result.json")
    # elif args.dataset == 'wmt':
    #     do_evaluate(args.fname, target_file="translation/result_files/gt/wmt_en_result")
    # else:
    #     print("Define a dataset to evaluate")