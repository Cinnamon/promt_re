from torch.utils.data import Dataset

import unicodedata


def normalize(text):
    return unicodedata.normalize('NFKC', text.strip().replace(' ', '').replace('\n', '').replace('\t', ''))


def read_file(filename):
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()
    lines = [x[:-1] for x in lines]  # remove \n
    i = 0
    samples = []
    while i < len(lines):
        items = lines[i].split(' ')
        sid = int(items[0])
        num = int(items[1])

        items = lines[i + 1].split(', ')
        pairs = []
        for pair in items:
            pair = pair.strip()
            ids = pair[1:-1].split(',')
            pairs.append((int(ids[0]), int(ids[1])))

        sents = []
        for j in range(i + 2, i + 2 + num):
            sents.append(lines[j].split(',')[-1])

        samples.append((sid, pairs, sents))
        i += 2 + num
    return samples


## for ECPE 10-fold, TransECPE 10-fold
def prepare_data(data, last_ans=True):
    ret = []
    for item in data:
        chunks = [normalize(x) for x in item[2]]
        cause_labels = set([x[0] - 1 for x in item[1]])  # to zero-based
        emotion_labels = set([x[1] - 1 for x in item[1]])
        text = '，'.join(chunks)
        if last_ans:
            cause = chunks[max(cause_labels)]
            emotion = chunks[max(emotion_labels)]
        else:
            cause = chunks[min(cause_labels)]
            emotion = chunks[min(emotion_labels)]
        # check
        if text.find(cause) == -1 or text.find(emotion) == -1:
            print('error', item)

        ret.append({'text': text, 'cause': cause, 'emotion': emotion})
    return ret


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
        data = read_file(data_path)
        data = prepare_data(data)
        texts = []
        targets = []
        for sample in data:
            texts.append(sample['text'])
            target = sample['text'].replace(sample['cause'], f'[{sample["cause"]}|cause]')
            target = target.replace(sample['emotion'], f'[{sample["emotion"]}|emotion]')
            targets.append(target)
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
