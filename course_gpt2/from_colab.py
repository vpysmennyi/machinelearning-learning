
from google.colab import drive
import time
import datetime

import seaborn as sns
import pandas as pd
import nltk
import numpy as np
import random

import torch
from torch import nn

import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, random_split

import torch_xla
import torch_xla.core.xla_model as xm

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup

class ArxivDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=768):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_msk = []

        for dat in data_list:
            tokenizer_encodings = tokenizer(dat, truncation=True, max_length=max_length, padding='max_length')

            self.input_ids.append(torch.tensor(tokenizer_encodings['input_ids']))
            self.attention_msk.append(torch.tensor(tokenizer_encodings['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_msk[idx]


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def get_distributed_sampler(dataset, shuffle=True):
    sampler = DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=shuffle)
    return sampler

def generate_sample(model, tokenizer):
    print('Generating sample:')
    # moving to cpu for generation
    model.to('cpu')
    sample_output = model.generate(bos_token_id=random.randint(1, 30000),
                                   do_sample=True,
                                   top_k=50,
                                   max_length=200,
                                   top_p=0.95,
                                   num_return_sequences=1)
    for i, sample in enumerate(sample_output):
        print(f'{i} : {tokenizer.decode(sample, skip_special_tokens=True)}')


def _run(model, tokenizer, dataset, tpu_dev):
    batch_size = 2

    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds,
                          sampler=get_distributed_sampler(train_ds ,shuffle=True),
                          batch_size=batch_size)
    val_dl = DataLoader(val_ds,
                        sampler=get_distributed_sampler(val_ds ,shuffle=False),
                        batch_size=batch_size)

    model.to(tpu_dev)

    epochs = 5
    learning_rate = 5e-4 * xm.xrt_world_size()
    warmup_steps = 1e2
    epsilon = 1e-8

    # this produces sample output every 100 steps
    sample_every = 100

    opt = AdamW(model.parameters(), lr=learning_rate)

    total_steps = int(len(train_ds )/ batch_size / xm.xrt_world_size() * epochs)

    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    train_loss = 0
    t0 = time.time()
    xm.master_print(f'num_train_steps = {total_steps}, world_size={xm.xrt_world_size()}')

    for epoch in range(epochs):

        p_train_dl = pl.ParallelLoader(train_dl, [tpu_dev])
        p_val_dl = pl.ParallelLoader(val_dl, [tpu_dev])

        t0_epoch = time.time()
        epoch_loss, val_epoch_loss = [], []

        model.train()
        for step, batch in enumerate(p_train_dl.per_device_loader(tpu_dev)):

            opt.zero_grad()

            b_input_ids = batch[0].to(tpu_dev)
            b_labels = batch[0].to(tpu_dev)
            b_att_mask = batch[1].to(tpu_dev)

            output = model(b_input_ids,
                           labels=b_labels,
                           attention_mask=b_att_mask,
                           token_type_ids=None)

            loss = output[0]
            epoch_loss.append(loss.detach().cpu())
            print(f'Step: {step}')
            if step % sample_every == 0 and step > 0:
                model.eval()
                generate_sample(model, tokenizer)
                # moving back to tpu to proceed with training
                model.to(tpu_dev)
                model.train()

            loss.backward()

            # need to use this for parallelism
            xm.optimizer_step(opt)
            scheduler.step()

        if epoch_loss:
            print \
                (f'TRAIN | Epoch {epoch} : Loss = {torch.stack(epoch_loss).detach().cpu().mean()}. Spent: {format_time(time.time() - t0_epoch)}')
        else:
            print('Epoch loss is empty')

        model.eval()
        for batch in p_val_dl.per_device_loader(tpu_dev):

            b_input_ids = batch[0].to(tpu_dev)
            b_labels = batch[0].to(tpu_dev)
            b_att_mask = batch[1].to(tpu_dev)

            output = model(b_input_ids,
                           labels=b_labels,
                           attention_mask=b_att_mask,
                           token_type_ids=None)

            loss = output[0]
            val_epoch_loss.append(loss)

        if val_epoch_loss:
            print(f'VAL | Epoch {epoch} : Loss = {torch.stack(val_epoch_loss).detach().cpu().mean()}')
        else:
            print('Empty val_epoch_loss')

    print(f'Total elapsed: {format_time(time.time() - t0)}')

def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    # initializing TPU device
    device = xm.xla_device()
    _run(mdl ,tokenizer ,dataset ,device)

drive.mount('/content/drive', force_remount=True)
nltk.download('punkt')
df = pd.read_json('drive/MyDrive/ArxivDS/arxiv-abstracts.json', lines=True, nrows=1000)

abstracts = df.abstract.copy()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<BOS>', eos_token='<EOS>', pad_token='<PAD>')
gpt2_config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
mdl = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt2_config)
mdl.resize_token_embeddings(len(tokenizer))
dataset = ArxivDataset(abstracts, tokenizer, max_length=512)
FLAG S ={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=1, start_method='fork')
