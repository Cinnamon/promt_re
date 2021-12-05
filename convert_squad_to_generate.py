import json

inf = 'D:\\promt_re\\Ueno\\test_0.json'
outf = 'D:\\promt_re\\Ueno\\test.json'

with open(inf, 'r', encoding='utf-8') as f:
    data = json.load(f)['data']

out_data = []

for sample in data:
    for paragraph in sample['paragraphs']:
        text = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            answer = qa['answers'][0]['text']
            out_data.append((text + ' . ' + question, answer))

with open(outf, 'w', encoding='utf-8') as f:
    json.dump(out_data, f, ensure_ascii=False)
