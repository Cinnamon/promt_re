import json

from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
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


def train_model(model, data_loader, test_dataloader, num_epochs, pred_file, tokenizer, parsing_function):
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
        epoch_loss = []
        for i, batch in loop:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            output = run_model(model, inputs, labels)
            loss = output[0]
            loss.backward()
            optim.step()
            model.zero_grad()
            lr_scheduler.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=round(float(loss), 5))
            epoch_loss.append(loss.data)
        print(f" average_loss: {sum(epoch_loss) / len(epoch_loss)}")
        if epoch % EVAL_EVERY == 0:
            evaluate(model, test_dataloader, pred_file, tokenizer, parsing_function)


def evaluate(model, data_loader, output_file, tokenizer, parsing_function):
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
    from models.validate import calculate_f1
    calculate_f1(output, parsing_function)
    json.dump(output, open(output_file, 'w', encoding='utf-8'))
