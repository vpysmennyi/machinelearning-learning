import os
import time
import datetime
import json

import pandas as pd
import numpy as np
import random

import torch

import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utilsgrep

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, random_split

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup

# set to True when running in google colab
in_colab = True


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


class ArxivAbstractGen():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        self.dataset = ArxivDataset(abstracts, tokenizer, max_length=CONFIG['encoding_max_length'])

        self.epochs = CONFIG['num_epochs']
        self.learning_rate = CONFIG['lr']
        self.warmup_steps = CONFIG['warmup_steps']
        self.epsilon = CONFIG['epsilon']
        self.batch_size = CONFIG['train_batch_size']

        self.train_ds_len = 0

    def get_model(self):
        return self.model

    def prepare_data_loader(self, ds):
        def _get_distributed_sampler(sliced_ds, shuffle=True):
            sampler = DistributedSampler(
                sliced_ds,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=shuffle)
            return sampler

        train_size = int(len(ds) * CONFIG['train_size_percent'] / 100)
        val_size = len(ds) - train_size

        train_ds, val_ds = random_split(ds, [train_size, val_size])

        self.train_ds_len = len(train_ds)

        train_dl = DataLoader(train_ds,
                              sampler=_get_distributed_sampler(train_ds, shuffle=True),
                              batch_size=self.batch_size)
        val_dl = DataLoader(val_ds,
                            sampler=_get_distributed_sampler(val_ds, shuffle=False),
                            batch_size=self.batch_size)
        
        return train_dl, val_dl

    def model_save(self):
        xm.master_print('saving the model')
        self.model.to('cpu')
        model_to_save = self.model.module if hasattr(self.model,
                                                'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(CONFIG['save_dir'])
        tokenizer.save_pretrained(CONFIG['save_dir'])
        xm.master_print('saved')

    def print_train_stats(self, stats):
        pd.set_option('precision', 2)
        stat = pd.DataFrame(data=stats)

        stat = stat.set_index('epoch')
        xm.master_print(stat)

    def generate_sample(self, top_k=50, max_l=200, top_p=0.92, num_seq=1):
        xm.master_print('Generating sample:')
        xm.rendezvous('generate smpl')
        # moving to cpu for generation
        self.model.to('cpu')
        sample_output = self.model.generate(bos_token_id=random.randint(1, 30000),
                                       do_sample=True,
                                       top_k=top_k,
                                       max_length=max_l,
                                       top_p=top_p,
                                       num_return_sequences=num_seq)
        for i, sample in enumerate(sample_output):
            xm.master_print(f'{i} : {self.tokenizer.decode(sample, skip_special_tokens=True)}')
        
    def run_training(self, tpu_dev):

        # get data loader sets
        train_dl, val_dl = self.prepare_data_loader(self.dataset)

        self.model.to(tpu_dev)

        opt = AdamW(self.model.parameters(), 
                    lr=self.learning_rate * xm.xrt_world_size(), 
                    eps=self.epsilon)

        total_steps = int(self.train_ds_len / self.batch_size / xm.xrt_world_size() * self.epochs)

        scheduler = get_linear_schedule_with_warmup(opt, 
                                                    num_warmup_steps=self.warmup_steps, 
                                                    num_training_steps=total_steps)

        t0 = time.time()
        xm.master_print(f'num_train_steps = {total_steps}, TPU cores={xm.xrt_world_size()}')

        train_stats = []

        for epoch in range(self.epochs):
            t0_epoch = time.time()
            
            p_train_dl = pl.ParallelLoader(train_dl, [tpu_dev])
            p_val_dl = pl.ParallelLoader(val_dl, [tpu_dev])

            val_epoch_loss = []
            epoch_loss = []

            ############ TRAINING
            self.model.train()
            for step, batch in enumerate(p_train_dl.per_device_loader(tpu_dev)):

                opt.zero_grad()

                b_input_ids = batch[0].to(tpu_dev)
                b_labels = batch[0].to(tpu_dev)
                b_att_mask = batch[1].to(tpu_dev)

                output = self.model(b_input_ids,
                               labels=b_labels,
                               attention_mask=b_att_mask,
                               token_type_ids=None)

                loss = output[0]
                epoch_loss.append(loss.item())

                loss.backward()

                # need to use this for parallelism
                xm.optimizer_step(opt)
                
                scheduler.step()

            train_time = format_time(time.time() - t0_epoch)
            epoch_loss = np.array(epoch_loss).mean()

            if epoch_loss:
                xm.master_print(f'TRAIN | Epoch {epoch + 1} : Loss = {epoch_loss}. Elapsed: {train_time}')
            else:
                print('Epoch loss is empty')

            ############ VALIDATION
            self.model.eval()

            val_t0 = time.time()

            for val_step, batch in enumerate(p_val_dl.per_device_loader(tpu_dev)):
                #xm.master_print(f'Val step {val_step}')
                b_input_ids = batch[0].to(tpu_dev)
                b_labels = batch[0].to(tpu_dev)
                b_att_mask = batch[1].to(tpu_dev)

                output = self.model(b_input_ids,
                               labels=b_labels,
                               attention_mask=b_att_mask,
                               token_type_ids=None)

                loss = output[0]
                val_epoch_loss.append(loss.item())

            val_time = format_time(time.time() - val_t0)
            val_epoch_loss = np.array(val_epoch_loss).mean()

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

            xm.save(self.model.state_dict(), CONFIG['save_dir'] + 'model_e.pt')

        xm.master_print(f'Total elapsed: {format_time(time.time() - t0)}')
        self.print_train_stats(train_stats)
        xm.rendezvous('leave')


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))


def _mp_fn(rank, flags, trainer_obj):
    torch.set_default_tensor_type('torch.FloatTensor')

    # initializing TPU device
    device = xm.xla_device()
    xm.rendezvous('init')
    trainer_obj.run_training(device)


# reading config
conf_file = './config.json'

if in_colab:
    from google.colab import drive
    drive.mount('/content/drive')
    conf_file = 'drive/MyDrive/ArxivDS/colab_config.json'

with open(conf_file) as f:
    CONFIG = json.load(f)

print('Execution configuration:')
for c in CONFIG:
    print(f'{c}' + ' ' * (30-len(c)) + f'{CONFIG[c]}')

if not in_colab:
    # initialize TPU for XLA when running on server
    os.environ["XRT_TPU_CONFIG"] = "tpu_worker;0;" + CONFIG['tpu_ip_address'] + ":8470"

#reading dataset
df = pd.read_json(CONFIG['datafile'], lines=True, nrows=80000)
abstracts = df.abstract

#loading GPT2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<BOS>', eos_token='<EOS>', pad_token='<PAD>')
gpt2_config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
mdl = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt2_config)
mdl.resize_token_embeddings(len(tokenizer))

trainer = ArxivAbstractGen(mdl, tokenizer)

xmp.spawn(_mp_fn, args=(CONFIG, trainer), nprocs=CONFIG['num_tpu_cores'], start_method='fork')

trainer.model_save()

#generating samples with a tuned model
# NOTE! not working here when running on 8 TPU cores
trainer.generate_sample(top_k=CONFIG['decode_top_k'],
                        max_l=CONFIG['decode_max_length'],
                        top_p=CONFIG['decode_top_p'],
                        num_seq=CONFIG['decode_num_test_samples'])


