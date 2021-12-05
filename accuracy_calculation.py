import json

data_file = 'D:\\t5_cause_effect\\pred.json'

data = json.load(open(data_file, 'r', encoding='utf-8'))

labels = ['location', 'person', 'organization', 'miscellaneous']


def parse_sentence(sentence):
    entities = []
    text = sentence.split('</s>')[0]
    entity_parts = text.split(';')
    for part in entity_parts:
        text = part.split('is')[0].strip()
        label = part.split('is')[-1].strip()
        if label in labels:
            entities.append((text, label))
    entities = list(set(entities))
    return entities


if __name__ == '__main__':
    sentence = "EU is organization; German is miscellaneous; British is miscellaneous;</s> <pad> <pad> <pad>"
    entities = parse_sentence(sentence)
    pass
