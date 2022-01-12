import json

from transformers.models.t5 import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
# from re_dataset import T5REDataset, collate_fn
# from conll_t5_tanl_dataset import T5ConllDataset, collate_fn
from datasets.conll_promt_retrieval_dataset import T5ConllDataset, collate_fn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from functools import partial

EVAL_EVERY = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_model(model, inputs, labels, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
    output = model(**inputs, labels=labels['input_ids'])
    loss = output.loss
    return loss, output


def train_model(model, data_loader, test_dataloader, num_epochs, pred_file, tokenizer):
    model.to(device)
    # optim = torch.optim.SGD(model.parameters(), lr=2e-5)
    optim = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = num_epochs * len(data_loader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=1000,
        num_training_steps=num_training_steps)
    for epoch in range(1, num_epochs + 1):
        loop = tqdm(enumerate(data_loader))
        for i, batch in loop:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            output = run_model(model, inputs, labels)
            loss = output[0]
            loss.backward()
            optim.step()
            lr_scheduler.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=round(float(loss), 5))
        if epoch % EVAL_EVERY == 0:
            evaluate(model, test_dataloader, pred_file, tokenizer)


def evaluate(model, data_loader, output_file, tokenizer):
    model.eval()
    output = []
    # save to file
    print("evaluate ...")
    for batch in tqdm(data_loader):
        labels = batch[1]['input_ids'].tolist()
        inputs = batch[0]['input_ids'].to(device)
        greedy_output = model.generate(inputs, inputs, num_beams=5, early_stopping=True)
        preds = greedy_output.cpu().detach().tolist()
        for i in range(len(labels)):
            gt = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(labels[i]))
            pred = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(preds[i]))
            output.append((gt, pred))
    json.dump(output, open(output_file, 'w', encoding='utf-8'))


if __name__ == '__main__':
    print("load model ...")
    pretrained_model =  't5-base' #'megagonlabs/t5-base-japanese-web'
    # pretrained_model = 'google/mt5-small'  # 'megagonlabs/t5-base-japanese-web'
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
    print("training")
    # train_file = 'data_combine/fold1_train.txt'
    # test_file = 'data_combine/fold1_test.txt'
    # train_data_set = T5ECPEDataset(train_file)
    # test_data_set = T5ECPEDataset(test_file)
    train_file = 'conll2003/train.txt'
    test_file = 'conll2003/test.txt'
    train_data_set = T5ConllDataset(train_file)
    test_data_set = T5ConllDataset(test_file)
    train_dataloader = DataLoader(train_data_set, shuffle=True, collate_fn=partial(collate_fn, tokenizer), batch_size=4)
    test_dataloader = DataLoader(test_data_set, shuffle=False, collate_fn=partial(collate_fn, tokenizer), batch_size=4)
    # pred_file = 'pred_fold1.json'
    pred_file = 'pred_conll_retrieval.json'
    train_model(model, train_dataloader, test_dataloader, 5, pred_file, tokenizer)
    # evaluate(model, data_loader, storages, tokenizer)
