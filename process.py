# processing
import os
import pdb
import re
import time
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import json
from multiprocessing import Pool, cpu_count     # https://docs.python.org/3/library/multiprocessing.html
from collections import OrderedDict
from itertools import chain
from tqdm import tqdm
from functools import partial
import torch

class ProtoQAExample():
    def __init__(self,idx,question, answer):
        self.idx = idx # keep idx for evaluation
        self.question = question
        self.answer = answer

class ProtoQADataset(Dataset):
    def __init__(self, idxs, encodings, labels):
        self.idxs = idxs
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['idx'] = self.idxs[idx]
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.idxs)

class ProtoQAProcessor():
    def read_example(self, file_name, is_training):
        with open(file_name,'r',encoding="utf8") as fin:
            examples = []
            for line in fin:
                items = json.loads(line.strip())
                idx = items['metadata']['id']
                question = items['question']['original']
                question = self.transform_question(question)
                if is_training:
                    answers = list(items['answers']['raw'].keys())
                    for answer in answers:
                        examples.append(ProtoQAExample(idx, question, answer))
                else:
                    examples.append(ProtoQAExample(idx,question,None))
                
        return examples
    
    def transform_question(self, origin):
        '''
        > after having kids name something that happens that interrupts a couples alone time at night

        > after having kids one thing that happens that interrupts a couples alone time at night is

        '''
        question = origin.lower()
        question = question.replace('.', '')
        question = question.replace(':', '')
        question = question.replace('?', '')
        question = question.replace('someone', 'one person')
        question = question.replace('someplace', 'one place')
        transform_dict = {
            "name something": "one thing",
            'tell me something': 'one thing',
            'name a ': 'one ',
            "name an ": "one ",
            "name": "",
            "SW tell me a ": "one ",
            "SW tell me an ": "one ",
            "SW what": "one",
            "SW give me a ": "one ",
            "SW tell me ": "",
            "which": "one",
            "what": "one",
            "how can you tell": "one way to tell",
        }
        order = ['name something', 'tell me something', 'name a ', 'name an ', 'name',
            'SW tell me a ', 'SW tell me an ', 'SW what', 'SW give me a ', 'SW tell me ',
            'which', 'what', 'how can you tell']
        transform = OrderedDict.fromkeys(order)
        transform.update(transform_dict)

        for pattern, trans in transform.items():
            if pattern.startswith('SW') and pattern[3:] in question:
                question = question.replace(pattern[3:], trans)
                question += ' is'
                break
            elif pattern in question:
                question = question.replace(pattern, trans)
                question += ' is'
                break
        else:
            question = 'Q: '+question +'? A: '

        return question
    def convert(self, example, tokenizer,  max_q_len, max_a_len, is_training = True):
        '''
        convert one question answer pair
        '''
        # pdb.set_trace()
        idx = example.idx
        tokenized_q = tokenizer(
            example.question,
            add_special_tokens=False,
            padding = 'max_length', 
            truncation = True,
            max_length = max_q_len,
            )
        
        if is_training:
            tokenized_a = tokenizer(
                example.answer,
                add_special_tokens=False,
                padding = 'max_length', 
                truncation = True,
                max_length = max_a_len, # atleast gpt2 don't have a cls
                )
            input_ids = tokenized_q["input_ids"] + tokenized_a["input_ids"][1:]
            labels = [ids if ids != tokenizer.pad_token_id else -100 for ids in input_ids]
            labels[:max_q_len] = [-100] * max_q_len

        else:
            input_ids = tokenized_q["input_ids"]
            labels = None
            
        
        # pdb.set_trace()

        return (idx, input_ids, labels)

    def convert_examples_to_features(self, tokenizer ,examples, max_q_len, max_a_len, is_training):
        encodings = {}
        encodings["input_ids"] = []
        labels = []
        tokenized = []
        # for idx, example in tqdm(enumerate(examples)):
        #     tokenized.append(self.convert(example, tokenizer, max_q_len, max_a_len))
        with Pool(processes=min(8, cpu_count())) as pool:
            annotate_ = partial(
                self.convert,
                tokenizer = tokenizer,
                max_q_len = max_q_len,
                max_a_len = max_a_len,
                is_training = is_training
            )

            tokenized = list(
                tqdm(
                    pool.imap(annotate_,examples,chunksize = 64),
                    total = len(examples),
                    desc = "convert protoqa examples to dataset"
                )
            )
        # pdb.set_trace()
        idxs = [item[0] for item in tokenized]
        encodings["input_ids"] = [item[1] for item in tokenized]
        if is_training:
            labels = [item[2] for item in tokenized]
            dataset = ProtoQADataset(idxs, encodings, labels)
        else: 
            dataset = ProtoQADataset(idxs, encodings, None)

        return dataset
def load_dataset(args, tokenizer, mode):
    processor = ProtoQAProcessor()
    is_training = True
    cache_dir = os.path.join(args.data_dir, "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_path = os.path.join(cache_dir, "{}_ql-{}_al-{}.cache".format(mode,str(args.max_q_len),str(args.max_a_len)))
    # load processed cache
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            dataset = torch.load(f)
        return dataset
        # torch.load(f)
    if mode == "train":
        file_path = os.path.join(args.data_dir, args.train_file)
        is_training = True
    elif mode == "dev" or mode == "valid":
        file_path = os.path.join(args.data_dir, args.dev_file)
    else:
        file_path = os.path.join(args.data_dir, args.test_file)
        is_training = False
    print(args.data_dir, file_path)

    examples = processor.read_example(file_path, is_training)
    dataset = processor.convert_examples_to_features(tokenizer, examples, args.max_q_len, args.max_a_len, is_training)
    torch.save(dataset,cache_path)
    return dataset

        