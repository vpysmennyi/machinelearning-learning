import json
import re

def build_dataset(ds, dest_file):
    '''
    Preparing dataset in json format. Also adding specific BOS, EOS tokens
    '''
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


json_file = open('arxiv-metadata-oai-snapshot.json')
build_dataset(json_file,'arxiv-abstracts.json')