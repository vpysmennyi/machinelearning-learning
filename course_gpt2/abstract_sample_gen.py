from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import random
import json

'''
Generating and printing text samples based on trained GPT2 model
'''

conf_file = './config.json'

with open(conf_file) as f:
    config = json.load(f)

tokenizer = GPT2Tokenizer.from_pretrained(config['save_dir'])
gpt2_config = GPT2Config.from_pretrained(config['save_dir'], output_hidden_states=False)
mdl = GPT2LMHeadModel.from_pretrained(config['save_dir'], config=gpt2_config)

sample_output = mdl.generate(bos_token_id=random.randint(1, 30000),
                                        do_sample=True,
                                        top_k=config['decode_top_k'],
                                        max_length=config['decode_max_length'],
                                        top_p=config['decode_top_p'],
                                        temperature=0.8,
                                        num_return_sequences=config['decode_num_test_samples'])
for i, sample in enumerate(sample_output):
    print(f'{i} : {tokenizer.decode(sample, skip_special_tokens=True)}')