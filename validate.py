import json
import re

LABEL_DICT = {'LOC': 'location',
              'PER': 'person',
              'ORG': 'organization',
              'MISC': 'miscellaneous'}


def load_dataset(data_path):
    lines = open(data_path, 'r', encoding='utf-8').readlines()
    texts = []
    targets = []
    text = ''
    target = []
    entity = ''
    label = ''
    for line in lines:
        if line.startswith('-DOCSTART-'):
            continue
        if len(line.strip()) == 0:
            if len(text) > 0:
                if len(entity) > 0:
                    target.append((entity, LABEL_DICT[label]))
                texts.append(text.strip())
                targets.append(target)
            text = ''
            target = []
            label = ''
            entity = ''
        else:
            word, tag = line.split()[0], line.split()[-1]
            text += ' ' + word
            if tag.startswith('B') or tag == 'O':
                if len(entity) > 0:
                    target.append((entity, LABEL_DICT[label]))
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
            target.append((entity, LABEL_DICT[label]))
        texts.append(text.strip())
        targets.append(target)
    return texts, targets


def parse_result(input_file):
    data = json.load(open(input_file, 'r', encoding='utf-8'))
    preds = []
    for sample in data:
        sample = sample[0]
        sample = sample.split('</s>')[0]
        samples = sample.split('.')
        samples = [x.strip() for x in samples if len(x.strip()) > 0]
        samples = [(x.split('is')[0].strip(), x.split('is')[-1].strip()) for x in samples]
        preds.append(samples)
    return preds


def parse_tanl_result(input_file):
    data = json.load(open(input_file, 'r', encoding='utf-8'))
    preds = []
    for sample in data:
        sample = sample[0]
        sample = sample.split('</s>')[0]
        matches = re.findall("\[.*?\]", sample)
        matches = [match[1:-1] for match in matches]
        matches = [(x.split('|')[0].strip(), x.split('|')[-1].strip()) for x in matches]
        preds.append(matches)
    return preds


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
    f1 = 2 * p * r / (p + r)
    return f1


if __name__ == '__main__':
    file = 'D:\\promt_re\\predictions\\pred_conll_retrieval.json'
    tanl_file = 'D:\\promt_re\\predictions\\pred_conll_tanl.json'
    gt_file = 'D:\\promt_re\\conll2003\\test.txt'
    _, gts = load_dataset(gt_file)
    preds = parse_result(file)
    preds_tanl = parse_tanl_result(tanl_file)
    f1_retrieval = calculate_f1_conll(preds, gts)
    f1_tanl = calculate_f1_conll(preds_tanl, gts)
    print("retrieval: ", f1_retrieval)
    print("tanl: ", f1_tanl)
