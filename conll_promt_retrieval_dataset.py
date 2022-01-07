import sentence_transformers

from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import random

"""
Prompt:
"sentence: text
"entities: x is y, z is t <sep>    <- incontext examples
.....
sentence: text.
entities: 
"""


class T5ConllDataset(Dataset):
    def __init__(self, data_path, max_len=384):
        self.label_dict = {'LOC': 'location',
                           'PER': 'person',
                           'ORG': 'organization',
                           'MISC': 'miscellaneous'}
        self.texts, self.targets = self.load_dataset(data_path)
        self.max_len = max_len
        self.task_description = ' Find entities in the text: '
        self.sentence_prefix = ' Input sentence: '
        self.target_prefix = ' Entities: '
        self.sep = ' <sep> '

        self.construct_df()

        pass

    def load_dataset(self, data_path):
        lines = open(data_path, 'r', encoding='utf-8').readlines()
        texts = []
        targets = []
        text = ''
        target = ''
        entity = ''
        label = ''
        for line in lines:
            if line.startswith('-DOCSTART-'):
                continue
            if len(line.strip()) == 0:
                if len(text) > 0:
                    if len(entity) > 0:
                        target += f" {entity} is {self.label_dict[label]} ."  # ' ' + entity + ' is ' + self.label_dict[label] + ';'
                    texts.append(text.strip())
                    targets.append(target.strip())
                text = ''
                target = ''
                label = ''
                entity = ''
            else:
                word, tag = line.split()[0], line.split()[-1]
                text += ' ' + word
                if tag.startswith('B') or tag == 'O':
                    if len(entity) > 0:
                        target += f" {entity} is {self.label_dict[label]} ."  # ' ' + entity + ' is ' + self.label_dict[label] + ';'
                        entity = ''
                        label = ''
                    if tag.startswith('B'):
                        label = tag[2:]
                        entity = word
                else:
                    assert label != ''
                    entity += ' ' + word
        if len(text) > 0:
            if len(entity) > 0:
                target += f" {entity} is {self.label_dict[label]} ."  # ' ' + entity + ' is ' + self.label_dict[label] + ';'
            texts.append(text.strip())
            targets.append(target.strip())
        return texts, targets

    def construct_df(self):
        self.df = pd.DataFrame({'text': self.texts, 'target': self.targets})
        input_text = []

        for i in tqdm(range(self.df.shape[0])):
            text, target = self.df.iloc[i].text, self.df.iloc[i].target
            input_prompt = f" {self.sentence_prefix} {text} {self.target_prefix}"
            incontext_examples = ''
            for j in range(3):
                idx = random.randint(0, self.df.shape[0] - 1)
                if idx == i:
                    continue
                example_text, example_target = self.df.iloc[idx].text, self.df.iloc[idx].target
                incontext_example = f" {self.sentence_prefix} {example_text} {self.target_prefix} {example_target} {self.sep} "
                if len(self.target_prefix.split()) + len(incontext_example.split()) + len(
                        incontext_example.split()) + len(input_prompt.split()) <= self.max_len:
                    incontext_examples += incontext_example
                else:
                    break
            input_text.append(self.task_description + incontext_examples + input_prompt)
        self.df['input_text'] = input_text

        pass

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.df.iloc[idx]['input_text'], self.df.iloc[idx]['target']


def collate_fn(tokenizer, batches):
    texts = [x[0] for x in batches]
    targets = [x[1] for x in batches]
    texts_tokenized = tokenizer(texts, return_tensors='pt',
                                padding='longest')
    targets_tokenized = tokenizer(targets, return_tensors='pt',
                                  padding='longest')
    return texts_tokenized, targets_tokenized


if __name__ == '__main__':
    file = 'D:\\promt_re\\conll2003\\train.txt'
    data_set = T5ConllDataset(file)
    (text, target) = data_set[0]
    print(text)
    print(target)
    pass
