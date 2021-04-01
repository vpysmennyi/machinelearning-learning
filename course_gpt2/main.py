
import torch
import pandas as pd
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ArxivDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        pass

#tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<BOS>', eos_token='<EOS>', pad_token='<PAD>')

df = pd.read_json('arxiv-metadata-oai-snapshot.json', lines=True)
abstracts = df.abstracts.copy()
print(abstracts)