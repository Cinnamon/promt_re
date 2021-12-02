from torch.utils.data import Dataset

class T5ConllDataset(Dataset):
    def __init__(self, data_path, max_len=128):
        self.label_dict = {'LOC': 'location',
                           'PER': 'person',
                           'ORG': 'organization',
                           'MISC': 'miscellaneous'}
        self.texts, self.targets = self.load_dataset(data_path)
        self.max_len = max_len
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
                        target += ' ' + entity + ' is ' + self.label_dict[label] + ';'
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
                        target += ' ' + entity + ' is ' + self.label_dict[label] + ';'
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
                target += ' ' + entity + ' is ' + self.label_dict[label] + ';'
            texts.append(text.strip())
            targets.append(target.strip())
        return texts, targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx]


def collate_fn(tokenizer, batches):
    texts = [x[0] for x in batches]
    targets = [x[1] for x in batches]
    texts_tokenized = tokenizer(texts, return_tensors='pt',
                                padding='longest')
    targets_tokenized = tokenizer(targets, return_tensors='pt',
                                  padding='longest')
    return texts_tokenized, targets_tokenized
