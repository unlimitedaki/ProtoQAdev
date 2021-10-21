# standard libraries
import os 
import sys
import json
import logging
import argparse
import time
import pdb
import random
import argparse
# third-part libraries
import numpy as np
# from apex import amp
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, DistributedSampler
import transformers
from transformers import GPT2Tokenizer
from transformers import AdamW,get_linear_schedule_with_warmup
from transformers.modeling_utils import SequenceSummary
from transformers import TrainingArguments
from transformers import Trainer

from process import *
from model import *
# if __name__ == "__main__":


def train(args):
# set up output_dir
    output_dir = os.path.join(args.output_dir,args.save_model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save all the args
    arg_dict = args.__dict__
    with open(os.path.join(output_dir,"args.json"),'w',encoding='utf8') as f:
        json.dump(arg_dict,f,indent=2,ensure_ascii=False)
    # setup logging

    # freeze seed
    if args.seed:
        set_seed(args)
    # processing
    # tokenizer = select_tokenizer(args)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = load_dataset(args, tokenizer, "train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler = train_sampler, batch_size = args.train_batch_size)
    if args.vaild_during_training:
        dev_dataset = load_dataset(args, tokenizer, "dev")
        dev_sampler = RandomSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler = dev_sampler, batch_size = args.eval_batch_size)
    use_trainer= True

    # load model
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # train !
    # if use_trainer:
    training_args = TrainingArguments(
    output_dir = "/content/output/",
    per_device_train_batch_size = 24,
    per_device_eval_batch_size = 32,
    learning_rate = args.learning_rate,
    num_train_epochs = 1,
    fp16 = True,
    log_level = "info",
    logging_steps=100,
    evaluation_strategy="epoch"
    )

    trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = dev_dataset,
    )
    trainer.train()

def eval(args,model=None):
    output_dir = os.path.join(args.output_dir,args.save_model_name)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dev_dataset = load_dataset(args, tokenizer, "test")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pred_path = os.path.join(output_dir, "predictions.jsonl")
    predictions = []
    if not model:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        device = torch.device("cuda:0")
        model = model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    for example in tqdm(dev_dataset, total= len(dev_dataset)):
        
        input = [example["input_ids"][:args.max_q_len,].tolist()]
        input = torch.tensor(input).to(device)
        generated = model.generate(
            input,
            max_length = args.max_q_len + args.max_a_len,
            do_sample = True,
            repetition_penalty=1,
            length_penalty = 0.1,
            num_return_sequences = 10
        )
        # pdb.set_trace()
        # question = tokenizer.decode(example["input_ids"][:40,].tolist(),clean_up_tokenization_spaces=True)
        res = {example['idx']:[]}

        for p in generated:
            answer = p[args.max_q_len:]
            answer = tokenizer.decode(answer.tolist(),clean_up_tokenization_spaces=True)
            res[example['idx']].append(answer)
        predictions.append(res)
    with open(pred_path,'w',encoding="utf8") as f:
        for p in predictions:
            json.dump(p, f, ensure_ascii = False)
            f.write('\n')

# def score()
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument("--data_dir",type = str,default = "/content/ProtoPrompt/protoqa-data/data")
    parser.add_argument("--test_file",type= str,default = "dev/dev.scraped.jsonl")
    parser.add_argument("--dev_file",type= str,default = "dev/dev.scraped.jsonl")
    parser.add_argument("--train_file",type = str,default = "train/train.jsonl")
    parser.add_argument("--output_dir",type = str,default = "model")
    parser.add_argument("--save_model_name",type = str,default = "GPT2-baseline")
    parser.add_argument("--tokenizer_name_or_path",type = str,default = "bert-base-cased")
    parser.add_argument("--origin_model",type = str,default = "gpt2", help = "origin model dir for training")

    # hyper parameters
    # parser.add_argument("--max_seq_length",type=int,default = 80 )
    parser.add_argument("--max_q_len", type = int, default = 40)
    parser.add_argument("--max_a_len", type = int, default = 40)

    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs",default=5,type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--train_batch_size", default=15, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=6, type=int, help="Batch size for eval.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--logging_step", default = 100, type = int, help = "steps for logging")
    # settings
    parser.add_argument("--n_gpu",type=int , default = 1)
    parser.add_argument("--fp16",action = "store_true")
    parser.add_argument("--save_method",type = str,default = "Best_Current")
    parser.add_argument("--do_finetune",action = "store_true",default = False)
    parser.add_argument("--seed",type = int,default = None,help = "freeze seed")
    parser.add_argument("--test",action = "store_true")
    parser.add_argument("--dev",action = "store_true")
    parser.add_argument("--vaild_during_training",action = "store_true",default = True)

    args = parser.parse_args(args = [])
    # train(args)
    eval(args)