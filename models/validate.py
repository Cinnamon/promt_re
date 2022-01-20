import json
import re

LABEL_DICT = {'LOC': 'location',
              'PER': 'person',
              'ORG': 'organization',
              'MISC': 'miscellaneous'}


def parse_conll_text(sample):
    sample = sample.split('</s>')[0]
    samples = sample.split('.')
    samples = [x.strip() for x in samples if len(x.strip()) > 0]
    samples = [(x.split('is')[0].strip(), x.split('is')[-1].strip()) for x in samples]
    return samples


def parse_conll_result(input_file):
    data = json.load(open(input_file, 'r', encoding='utf-8'))
    gts = []
    preds = []
    for sample in data:
        gts.append(parse_conll_text(sample[0]))
        preds.append(parse_conll_text(sample[1]))
    return preds, gts


def parse_tanl_text(sample):
    sample = sample.split('</s>')[0]
    matches = re.findall("\[.*?\]", sample)
    matches = [match[1:-1] for match in matches]
    matches = [(x.split('|')[0].strip(), x.split('|')[-1].strip()) for x in matches]
    return matches


def parse_tanl_result(data):
    gts = []
    preds = []
    for sample in data:
        gts.append(parse_tanl_result(sample[0]))
        preds.append(parse_tanl_text(sample[1]))
    return preds, gts


def calculate_f1(prediction_output, parsing_function):
    preds, gts = parsing_function(prediction_output)
    f1 = calculate_f1_conll(preds, gts)
    print("EVALUATION F1: ", f1)
    return f1


def calculate_f1_conll(preds, gts):
    tp = 0
    fn = 0
    fp = 0
    for i in range(len(preds)):
        pred_set = set(preds[i])
        gt_set = set(gts[i])
        common = pred_set.intersection(gt_set)
        tp += len(common)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    if p == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)
    return f1

