
import time
import datetime
import json

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

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup

# setup colab
in_colab = False


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

def print_train_stats(stats):
    pd.set_option('precision', 2)
    stat = pd.DataFrame(data=stats)

    stat = stat.set_index('epoch')
    xm.master_print(stat)

def generate_sample(model, tokenizer, top_k = 50, max_l = 200, top_p = 0.92, num_seq=1):
    xm.master_print('Generating sample:')
    # moving to cpu for generation
    model.to('cpu')
    sample_output = model.generate(bos_token_id=random.randint(1, 30000),
                                   do_sample=True,
                                   top_k=top_k,
                                   max_length=max_l,
                                   top_p=top_p,
                                   num_return_sequences=num_seq)
    for i, sample in enumerate(sample_output):
        xm.master_print(f'{i} : {tokenizer.decode(sample, skip_special_tokens=True)}')


def _run(model, tokenizer, dataset, tpu_dev):
    batch_size = config['train_batch_size']

    train_size = int(len(dataset) * config['train_size_percent']/100)
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds,
                          sampler=get_distributed_sampler(train_ds ,shuffle=True),
                          batch_size=batch_size)
    val_dl = DataLoader(val_ds,
                        sampler=get_distributed_sampler(val_ds ,shuffle=False),
                        batch_size=batch_size)

    model.to(tpu_dev)
    devices = xm.get_xla_supported_devices(max_devices=config['num_tpu_cores'])
    model = dp.DataParallel(model, device_ids=devices)

    epochs = config['num_epochs']
    learning_rate = config['lr'] * xm.xrt_world_size()
    warmup_steps = config['warmup_steps']
    epsilon = config['epsilon']

    opt = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)

    total_steps = int(len(train_ds )/ batch_size / xm.xrt_world_size() * epochs)

    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    t0 = time.time()
    xm.master_print(f'num_train_steps = {total_steps}, world_size={xm.xrt_world_size()}')

    train_stats = []

    for epoch in range(epochs):

        p_train_dl = pl.ParallelLoader(train_dl, [tpu_dev])
        p_val_dl = pl.ParallelLoader(val_dl, [tpu_dev])

        t0_epoch = time.time()
        val_epoch_loss = 0
        epoch_loss = 0

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
            epoch_loss += loss.item()

            if step % 10 ==0:
                xm.master_print(f'Step: {step}')

            if config['mid_sample_enable']:
                if step % config['sample_every_steps'] == 0 and step > 0:
                    model.eval()

                    generate_sample(model, tokenizer, config['decode_top_k'],config['decode_max_length'], config['decode_top_p'])
                    #moving back to tpu to proceed with training
                    model.to(tpu_dev)
                    model.train()

            loss.backward()

            # need to use this for parallelism
            xm.optimizer_step(opt)
            scheduler.step()

        train_time = format_time(time.time() - t0_epoch)
        epoch_loss = epoch_loss /len(train_dl)

        if epoch_loss:
            xm.master_print(f'TRAIN | Epoch {epoch + 1} : Loss = {epoch_loss}. Elapsed: {train_time}')
        else:
            print('Epoch loss is empty')

        model.eval()

        val_t0 = time.time()

        for val_step, batch in enumerate(p_val_dl.per_device_loader(tpu_dev)):
            xm.master_print(f'Val step {val_step}')
            b_input_ids = batch[0].to(tpu_dev)
            b_labels = batch[0].to(tpu_dev)
            b_att_mask = batch[1].to(tpu_dev)

            output = model(b_input_ids,
                           labels=b_labels,
                           attention_mask=b_att_mask,
                           token_type_ids=None)

            loss = output[0]
            val_epoch_loss += loss.item()

        val_time = format_time(time.time() - val_t0)
        val_epoch_loss = val_epoch_loss /len(val_dl)

        if val_epoch_loss:
            xm.master_print(f'VAL | Epoch {epoch + 1} : Loss = {val_epoch_loss}. Elapsed: {val_time}')
        else:
            print('Empty val_epoch_loss')

        # Record all statistics from this epoch.
        train_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': epoch_loss,
                'Valid. Loss': val_epoch_loss,
                'Training Time': train_time,
                'Validation Time': val_time
            }
        )

        xm.master_print('saving the model')
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(config['save_dir'])
        tokenizer.save_pretrained(config['save_dir'])
        xm.master_print('saved')
    xm.master_print(f'Total elapsed: {format_time(time.time() - t0)}')
    print_train_stats(train_stats)

    model.to('cpu')

    # final result
    xm.master_print('Testing the model')
    generate_sample(model, tokenizer, num_seq=config['decode_num_test_samples'])

def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')

    # initializing TPU device
    device = xm.xla_device()
    _run(mdl ,tokenizer ,dataset ,device)

nltk.download('punkt')

# reading config
conf_file = './config.json'

if in_colab:
    from google.colab import drive
    drive.mount('/content/drive')
    conf_file = 'drive/MyDrive/ArxivDS/colab_config.json'

with open(conf_file) as f:
    config = json.load(f)

print('Execution configuration:')
for c in config:
    print(f'{c}' + ' '*(30-len(c)) + f'{config[c]}')

#reading dataset
df = pd.read_json(config['datafile'], lines=True, nrows=2000)
abstracts = df.abstract

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<BOS>', eos_token='<EOS>', pad_token='<PAD>')
gpt2_config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
mdl = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt2_config)
mdl.resize_token_embeddings(len(tokenizer))
dataset = ArxivDataset(abstracts, tokenizer, max_length=config['encoding_max_length'])
FLAGS ={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=config['num_tpu_cores'], start_method='fork')

#export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
