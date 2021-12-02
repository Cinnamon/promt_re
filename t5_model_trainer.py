import json

from transformers.models.mt5 import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from conll_t5_dataset import T5ConllDataset, collate_fn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from functools import partial

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
            loss = run_model(model, inputs, labels)
            loss.backward()
            optim.step()
            lr_scheduler.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=round(float(loss), 5))
        evaluate(model, test_dataloader, pred_file, tokenizer)


def evaluate(model, data_loader, output_file, tokenizer):
    model.eval()
    output = []
    # save to file
    for batch in data_loader:
        labels = batch[1]['input_ids'].tolist()
        inputs = batch[0]['input_ids'].to(device)
        greedy_output = model.generate(inputs, max_length=50)
        preds = greedy_output.cpu().detach().tolist()
        for i in range(len(labels)):
            gt = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(labels[i]))
            pred = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(preds[i]))
            output.append((gt, pred))
    json.dump(output, open(output_file, 'w', encoding='utf-8'))


if __name__ == '__main__':
    print("load model ...")
    pretrained_model = 'google/mt5-base'
    model = MT5ForConditionalGeneration.from_pretrained(pretrained_model)
    tokenizer = MT5Tokenizer.from_pretrained(pretrained_model)
    print("training")
    train_file = 'conll2003\\test.txt'
    data_set = T5ConllDataset(train_file)
    train_dataloader = DataLoader(data_set, shuffle=True, collate_fn=partial(collate_fn, tokenizer), batch_size=1)
    test_dataloader = DataLoader(data_set, shuffle=False, collate_fn=partial(collate_fn, tokenizer), batch_size=1)
    pred_file = 'pred.json'
    train_model(model, train_dataloader, test_dataloader, 10, pred_file, tokenizer)
    # evaluate(model, data_loader, storages, tokenizer)
