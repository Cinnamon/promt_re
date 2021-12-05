import json

from torch.utils.data import Dataset

class T5REDataset(Dataset):
    def __init__(self, data_path, max_len= 512):
        self.texts, self.targets = self.load_dataset(data_path)
        self.max_len = max_len
        pass

    def load_dataset(self, data_path):
        data = json.load(open(data_path, 'r', encoding='utf-8'))
        texts = []
        targets = []
        for sample in data:
            texts.append(sample[0])
            targets.append(sample[1])
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
    print(targets)
    1/0
    targets_tokenized = tokenizer(targets, return_tensors='pt',
                                  padding='longest')
    return texts_tokenized, targets_tokenized
