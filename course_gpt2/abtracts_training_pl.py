import json
import pandas as pd
import numpy as np
import random
import torch
import pytorch_lightning as pyl
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, random_split

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import AdamW, get_cosine_schedule_with_warmup

# set this to True when running in google colab
in_colab = False

random.seed(42)


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


class ArxivAbstractGen(pyl.LightningModule):
    def __init__(self):
        super(ArxivAbstractGen, self).__init__()

        # loading GPT2
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<BOS>', eos_token='<EOS>', pad_token='<PAD>')
        gpt2_config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
        self.model = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt2_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.epochs = config['num_epochs']
        self.learning_rate = config['lr']
        self.epsilon = config['epsilon']
        self.batch_size = config['train_batch_size']

        self.train_ds_len = 0

        self.train_loss = []
        self.val_loss = []
        self.total_stats = []

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, b_input_ids, b_labels, b_att_mask):
        output = self.model(b_input_ids,
                            labels=b_labels,
                            attention_mask=b_att_mask,
                            token_type_ids=None)
        return output

    def configure_optimizers(self):
        train_steps = int((self.train_ds_len / self.batch_size / config['num_tpu_cores']) * self.epochs)

        opt = AdamW(self.model.parameters(),
                    lr=self.learning_rate,
                    eps=self.epsilon)

        sched = {'scheduler': get_cosine_schedule_with_warmup(opt,
                                                              num_warmup_steps=config['warmup_steps'],
                                                              num_training_steps=train_steps),
                 'interval': 'step'
                 }
        return [opt], [sched]

    def training_step(self, batch, index):
        b_input_ids = batch[0]
        b_labels = batch[0]
        b_att_mask = batch[1]

        output = self(b_input_ids,
                      b_labels,
                      b_att_mask)

        loss = output[0]
        self.train_loss.append(loss.item())
        # self.log('train_loss', loss.item(), logger=True)
        return loss

    def training_epoch_end(self, outputs):
        # calculate average training loss for epoch
        te_loss = np.array(self.train_loss).mean()
        d = {'train_loss': te_loss}

        # updating stats
        if len(self.total_stats) == self.current_epoch:
            self.total_stats.append(d)
        else:
            self.total_stats[self.current_epoch].update(d)

        self.print(f'Epoch {self.current_epoch}, train_loss: {te_loss}')

    def validation_step(self, batch, index):
        b_input_ids = batch[0]
        b_labels = batch[0]
        b_att_mask = batch[1]

        output = self(b_input_ids,
                      b_labels,
                      b_att_mask)

        loss = output[0]
        self.val_loss.append(loss.item())
        # self.log('val_loss', loss.item(), logger=True)
        return loss

    def validation_epoch_end(self, outputs):
        # calculating validation loss on epoch
        ve_loss = np.array(self.val_loss).mean()
        d = {'val_loss': ve_loss}

        # updating stats
        if len(self.total_stats) == self.current_epoch:
            self.total_stats.append(d)
        else:
            self.total_stats[self.current_epoch].update(d)

        self.print(f'Epoch {self.current_epoch}, val_loss: {ve_loss}')

    def model_save(self):
        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(config['save_dir'])
        self.tokenizer.save_pretrained(config['save_dir'])
        self.print('model saved')


def prepare_data_loader(ds):
    # dividing dataset to train and validation datasets with a given distribution scale
    train_size = int(len(ds) * config['train_size_percent'] / 100)
    val_size = len(ds) - train_size

    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_dl = DataLoader(train_ds,
                          sampler=RandomSampler(train_ds),
                          batch_size=config['train_batch_size'], num_workers=config['data_loader_workers'])
    val_dl = DataLoader(val_ds,
                        sampler=SequentialSampler(val_ds),
                        batch_size=config['train_batch_size'], num_workers=config['data_loader_workers'])
    return train_dl, val_dl, len(train_ds)


def generate_sample(model, tokenizer, top_k=50, max_l=200, top_p=0.92, num_seq=1):
    model.to('cpu')
    print('Generating sample:')
    # moving to cpu for generation
    sample_output = model.generate(bos_token_id=random.randint(1, 30000),
                                   do_sample=True,
                                   top_k=top_k,
                                   max_length=max_l,
                                   top_p=top_p,
                                   temperature=0.8,
                                   num_return_sequences=num_seq)
    for i, sample in enumerate(sample_output):
        print(f'{i} : {tokenizer.decode(sample, skip_special_tokens=True)}')


# reading config
conf_file = './config.json'

# define specific path for config file, when running in colab
if in_colab:
    from google.colab import drive

    drive.mount('/content/drive')
    conf_file = 'drive/MyDrive/ArxivDS/colab_config.json'

# loading training config
with open(conf_file) as f:
    config = json.load(f)

# printing config
print('Execution configuration:')
for c in config:
    print(f'{c}' + ' ' * (30 - len(c)) + f'{config[c]}')

# reading dataset from file
df = pd.read_json(config['datafile'], lines=True)
abstracts = df.abstract

lit_model = ArxivAbstractGen()

#preparing dataset
dataset = ArxivDataset(abstracts, lit_model.get_tokenizer(), max_length=config['encoding_max_length'])

train_dl, val_dl, len_train_ds = prepare_data_loader(dataset)
lit_model.train_ds_len = len_train_ds

trainer = pyl.Trainer(tpu_cores=config['num_tpu_cores'], max_epochs=config['num_epochs'])

trainer.fit(lit_model, train_dl, val_dl)

lit_model.model_save()

print(lit_model.total_stats)

generate_sample(lit_model.get_model(), lit_model.get_tokenizer(),
                top_k=config['decode_top_k'],
                max_l=config['decode_max_length'],
                top_p=config['decode_top_p'],
                num_seq=config['decode_num_test_samples'])

