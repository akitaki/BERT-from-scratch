import os
from pathlib import Path
import torch
import re
import random
import transformers, datasets
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import itertools
import math
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam


### STEP 0: PREPROCESSING ### 
MAX_LEN = 64

# Load from datasets
movie_convos = './datasets/movie_conversations.txt'
movie_lines = './datasets/movie_lines.txt'

with open(movie_convos, 'r', encoding='iso-8859-1') as c:
    convos = c.readlines()
with open(movie_lines, 'r', encoding='iso-8859-1') as l:
    lines = l.readlines()

# Extract and save the line IDs and corresponding lines
lines_dict = {}
for line in lines:
    strings = line.split(' +++$+++ ')
    lines_dict[strings[0]] = strings[-1]

# Create prompt-response pairs from convos, first line is prompt, next line is response.
pairs = []
for convo in convos:
    # line IDs are already written as a Python list in the dataset
    ids = eval(convo.split(' +++$+++ ')[-1])
    for i in range(len(ids)-1):
        pr_pair = []

        prompt = lines_dict[ids[i]].strip()
        response = lines_dict[ids[i+1]].strip()

        pr_pair.append(' '.join(prompt.split()[:MAX_LEN]))
        pr_pair.append(' '.join(response.split()[:MAX_LEN]))
        
        pairs.append(pr_pair)

# debug: check if pairs look right
# print(pairs[20])

### STEP 0.5: TOKENIZATION ###
# Special tokens: [CLS] [SEP] [PAD] [MASK] [UNK]
# WordPiece tokenizer

# Create text files from prompt-response pairs, each file is a batch of 10,000 chars
text_data = []
num_files = 0

for sample in tqdm([pair[0] for pair in pairs]):
    text_data.append(sample)

    if len(text_data) == 10000:
        with open(f'./data/text_{num_files}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        
        text_data = []
        num_files += 1

# store path to files
paths_list = [x.as_posix() for x in Path('./data').glob('*.txt')]

# Train the tokenizer
# BertWordPieceTokenizer binds to faster Rust implementation, potentially superseded by BertTokenizerFast from Transformers lib 
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=True
    )

tokenizer.train( 
    files = paths_list,
    vocab_size = 30000, 
    min_frequency = 5,
    limit_alphabet = 1000, 
    wordpieces_prefix='##',
    special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    )

tokenizer.save_model('./tokenizers', 'my-bert-tokenizer')
tokenizer = BertTokenizer.from_pretrained('./tokenizers/my-bert-tokenizer-vocab.txt')

### STEP 1: PRETRAINING ###
# Masked language prediction and next sequence prediction
# BERT dataset class with methods that implement these
class BERTDataset(Dataset):
    def __init__(self, data_pair, tokenizer, seq_len = MAX_LEN):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair

    def __getitem__(self, item):
        # Step 1: get random sentence pair, either negative or positive (saved as is_next_label)
        t1, t2, is_next_label = self.get_sent(item)

        # Step 2: replace random words in sentence with mask / random words
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # Step 3: Adding CLS and SEP tokens to the start and end of sentences
        # Adding PAD token for labels
        t1 = [self.tokenizer.vocab['[CLS]']] + t1_random + [self.tokenizer.vocab['[SEP]']]
        t2 = t2_random + [self.tokenizer.vocab['[SEP]']]
        t1_label = [self.tokenizer.vocab['[PAD]']] + t1_label + [self.tokenizer.vocab['[PAD]']]
        t2_label = t2_label + [self.tokenizer.vocab['[PAD]']]

        # Step 4: combine sentence 1 and 2 as one input
        # adding PAD tokens to make the sentence same length as seq_len
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input, "bert_label": bert_label, "segment_label": segment_label, "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def get_sent(self, index):
        '''return random sentence pair'''
        t1, t2 = self.get_corpus_line(index)

        # negative or positive pair, for next sentence prediction
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        '''return sentence pair'''
        return self.lines[item][0], self.lines[item][1]

    def get_random_line(self):
        '''return random single sentence'''
        return self.lines[random.randrange(len(self.lines))][1]
    
    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        output = []

        # 15% of the tokens would be replaced
        for i, token in enumerate(tokens):
            prob = random.random()

            # remove cls and sep token
            token_id = self.tokenizer(token)['input_ids'][1:-1]

            if prob < 0.15:
                prob /= 0.15

                # 80% chance change token to mask token
                if prob < 0.8:
                    for i in range(len(token_id)):
                        output.append(self.tokenizer.vocab['[MASK]'])

                # 10% chance change token to random token
                elif prob < 0.9:
                    for i in range(len(token_id)):
                        output.append(random.randrange(len(self.tokenizer.vocab)))

                # 10% chance change token to current token
                else:
                    output.append(token_id)

                output_label.append(token_id)

            else:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)

        # Flatten into one big output
        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        
        # Verify
        assert len(output) == len(output_label)
        
        return output, output_label
    

### STEP 2: EMBEDDING ###
# PyTorch NN layers for positional embedding and token, segment embeddings

# Using positional encoding scheme from "Attention is All You Need"
class PositionalEmbedding(torch.nn.module):
    def __init__(self, output_dim, max_len = 128):
        super().__init__()

        pe = torch.zeros(max_len, output_dim)
        pe.require_grad = False

        for pos in range(max_len):
            for i in range(0, output_dim, 2):
                pe[pos, i] = math.sin(pos / 10000**(2*i / output_dim))
                pe[pos, i+1] = math.cos(pos / 10000**(2*(i+1) / output_dim))

        # Add the batch dimension
        self.pe = pe.unsqueeze(dim=0)

    def forward(self, x):
        return self.pe

#  














