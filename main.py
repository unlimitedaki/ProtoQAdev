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
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, DistributedSampler
import transformers
from transformers import GPT2Tokenizer
from transformers import AdamW,get_linear_schedule_with_warmup
from transformers.modeling_utils import SequenceSummary
from transformers import TrainingArguments
from transformers import Trainer

from process import *
from model import *
from protoqa_evaluator.data_processing import load_question_answer_clusters_from_jsonl
from protoqa_evaluator.evaluation import multiple_evals
from protoqa_evaluator.common_evaluations import exact_match_all_eval_funcs
answers_dict = {}

# if __name__ == "__main__":


def train(args):
# set up output_dir
    print(args.data_dir)
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
    train_dataloader = DataLoader(train_dataset, sampler = train_sampler, batch_size = args.per_device_train_batch_size)
    if args.vaild_during_training:
        dev_dataset = load_dataset(args, tokenizer, "dev")
        dev_sampler = RandomSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler = dev_sampler, batch_size = args.per_device_eval_batch_size)
    use_trainer= True

    # load model
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # train !
    # if use_trainer:
    training_args = TrainingArguments(
    output_dir = output_dir,
    per_device_train_batch_size = args.per_device_train_batch_size,
    per_device_eval_batch_size = args.per_device_eval_batch_size,
    learning_rate = args.learning_rate,
    num_train_epochs = args.num_train_epochs,
    fp16 = True,
    log_level = "info",
    logging_steps=args.logging_steps,
    evaluation_strategy="epoch",
    save_strategy = "epoch",
    )

    trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = dev_dataset,
    )
    trainer.train()

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    # pdb.set_trace()
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(length):

            inputs = {'input_ids': generated}
            with autocast():
                outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated
from nltk.corpus import stopwords
def eval(args,model=None):
    output_dir = os.path.join(args.output_dir,args.save_model_name)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    en_stopwords = set(stopwords.words('english'))
    dev_dataset = load_dataset(args, tokenizer, "test")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # pred_path = os.path.join(output_dir, "predictions.jsonl")
    predictions = []
    if not model:
        check_point_path = os.path.join(output_dir,args.check_point_name)
        pred_path = os.path.join(check_point_path, "predictions.jsonl")
        model = GPT2LMHeadModel.from_pretrained(check_point_path)
        device = torch.device("cuda:0")
        model = model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    use_generate = False
    for example in tqdm(dev_dataset, total= len(dev_dataset)):
        
        
        if use_generate:
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
            res = {example['idx']:[]}

            for p in generated:
                answer = p[args.max_q_len:]
                answer = tokenizer.decode(answer.tolist(),clean_up_tokenization_spaces=True)
                res[example['idx']].append(answer)
        else:
            input = example["input_ids"][:args.max_q_len,].tolist()
            input = torch.tensor(input).to(device)
            generated = sample_sequence(
                model=model,
                context=input,
                num_samples=100,
                length=10,
                temperature=0.69,
                top_k=0,
                top_p=0.9,
                repetition_penalty=1,
                is_xlnet=False,
                is_xlm_mlm=False,
                xlm_mask_token=None,
                xlm_lang=None,
                device=device,
            )
            
            generated = generated[:, 40:].tolist()
            res = {example['idx']:[]}
            for o in generated:
                try:
                    text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                    text = text[: text.find(args.stop_token)+1 if args.stop_token else None]
                    text = text.strip()
                    if text.endswith('.'):
                        text = text[:-1]
                # print(text)
                    nostop_text_list = [tok for tok in text.split(' ') if tok not in en_stopwords]
                    nostop_text = " ".join(nostop_text_list)
                    res[example['idx']].append(nostop_text)
                except Exception as ex:
                    print(res)
                    continue
                # print(nostop_text)
                # if qidx[single_question_idx] not in prediced_dev:
                #     prediced_dev[qidx[single_question_idx]] = [nostop_text]
                # else:
                #     prediced_dev[qidx[single_question_idx]].append(nostop_text)
                # result.append((raw_text, nostop_text))
            # pdb.set_trace()
        
        # question = tokenizer.decode(example["input_ids"][:40,].tolist(),clean_up_tokenization_spaces=True)
        
        predictions.append(res)
    with open(pred_path,'w',encoding="utf8") as f:
        for p in predictions:
            json.dump(p, f, ensure_ascii = False)
            f.write('\n')
    with open(pred_path,'r') as f:
        for line in f:
            item = json.loads(line)
            for key in item.keys():
                answers_dict[key] = item[key]
    question_data = load_question_answer_clusters_from_jsonl(os.path.join(args.data_dir,args.test_file))
    res = multiple_evals(exact_match_all_eval_funcs, question_data, answers_dict=answers_dict)

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
    parser.add_argument("--check_point_name", type = str, help = "checkpoint for eval")
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
    parser.add_argument("--per_device_train_batch_size", default=15, type=int, help="Batch size for training.")
    parser.add_argument("--per_device_eval_batch_size", default=6, type=int, help="Batch size for eval.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--logging_steps", default = 100, type = int, help = "steps for logging")
    # settings
    parser.add_argument("--n_gpu",type=int , default = 1)
    parser.add_argument("--fp16",action = "store_true")
    parser.add_argument("--save_method",type = str,default = "Best_Current")
    parser.add_argument("--do_finetune",action = "store_true",default = False)
    parser.add_argument("--seed",type = int,default = None,help = "freeze seed")
    parser.add_argument("--test",action = "store_true")
    parser.add_argument("--dev",action = "store_true")
    parser.add_argument("--vaild_during_training",action = "store_true",default = True)
    parser.add_argument('--stop_token', type=str, default=".",
                        help="Token at which text generation is stopped")

    args = parser.parse_args()
    if args.test:
    # train(args)
        eval(args)
    else:
        train(args)