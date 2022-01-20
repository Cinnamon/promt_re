from transformers.models.t5 import T5ForConditionalGeneration, T5Tokenizer
from datasets.conll_t5_dataset import T5ConllDataset, collate_fn
from torch.utils.data import DataLoader
from functools import partial
from models.trainer import train_model
from models.validate import parse_conll_result

if __name__ == '__main__':
    print("load model ...")
    pretrained_model = 't5-base'
    from datasets import prj_path
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
    print("training")
    train_file = prj_path + '/conll2003/train.txt'
    test_file = prj_path + '/conll2003/test.txt'
    train_data_set = T5ConllDataset(train_file)
    test_data_set = T5ConllDataset(test_file)
    train_dataloader = DataLoader(train_data_set, shuffle=True, collate_fn=partial(collate_fn, tokenizer), batch_size=2)
    test_dataloader = DataLoader(test_data_set, shuffle=False, collate_fn=partial(collate_fn, tokenizer), batch_size=2)
    pred_file = 'pred_conll.json'
    train_model(model, train_dataloader, test_dataloader, 5, pred_file, tokenizer, parse_conll_result)
