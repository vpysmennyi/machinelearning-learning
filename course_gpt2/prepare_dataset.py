import json
import re

import seaborn as sns
import nltk
import numpy as np


def build_dataset(ds, dest_file):
    """
    Preparing dataset in json format. Also adding specific BOS, EOS tokens
    """
    abstracts = []
    id = 0
    for n in ds:
        abstract = dict()
        d = json.loads(n)
        bos_token = '<BOS>'
        eos_token = '<EOS>'
        abstract['id'] = id
        abstract['abstract'] = re.sub(r'\n', ' ', bos_token + ' ' + d['abstract'] + ' ' + eos_token)
        abstracts.append(abstract)
        id += 1

    with open(dest_file, 'w') as jw:
        for abstract in abstracts:
            json.dump(abstract, jw)
            jw.write('\n')

    return abstracts


#prepare dataset and save to file
json_file = open('arxiv-metadata-oai-snapshot.json')
dataset = build_dataset(json_file,'arxiv-abstracts.json')

abstr_lengths = []

# tokenizing dataset with nltk for further analysis
for dat in dataset:
    tokens = nltk.word_tokenize(dat['abstract'])

    abstr_lengths.append(len(tokens))

abstr_lengths = np.array(abstr_lengths)

print(np.average(abstr_lengths)) # average abstract length
print(np.max(abstr_lengths))     # max abstract length
print(len(abstr_lengths[abstr_lengths > 256])/len(abstr_lengths)) # percent of abstracts in dataset with len > 256 tokens

# printing distribution graph
sns.distplot(abstr_lengths)

