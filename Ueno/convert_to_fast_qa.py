import json
import pickle


def convert_ueno_to_fast_qa(file_path, output_folder):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    fast_qa_data = {'data': []}
    paragraphs = []
    for i, sample in enumerate(data):
        trouble_desc = sample[1]
        cause_desc = sample[2]
        hazards = sample[3]
        states = sample[4]
        effects = sample[5]
        context = trouble_desc + cause_desc
        hazard_answers = []
        state_answers = []
        effect_answers = []
        for hazard in hazards:
            start = context.find(hazard)
            if start != -1:
                # assert start != -1
                hazard_answers.append({"text": hazard, "answer_start": start})
        for state in states:
            start = context.find(state)
            if start != -1:
                # assert start != -1
                state_answers.append({"text": state, "answer_start": start})
        for effect in effects:
            start = context.find(effect)
            # assert start != -1, f"{context} || {effect}"
            if start != -1:
                effect_answers.append({"text": effect, "answer_start": start})
        hazard_qa = {'question': 'hazard',
                     'answers': hazard_answers,
                     'is_impossible': len(hazard_answers) == 0,
                     'id': str(i) + '.1'}
        state_qa = {'question': 'state',
                    'answers': state_answers,
                    'is_impossible': len(state_answers) == 0,
                    'id': str(i) + '.2'}
        effect_qa = {'question': 'effect',
                     'answers': effect_answers,
                     'is_impossible': len(effect_answers) == 0,
                     'id': str(i) + '.3'}
        qas = [hazard_qa, state_qa, effect_qa]
        paragraphs.append({'context': context,
                           'qas': qas})
    train_paragraphs = paragraphs[:int(0.7 * len(paragraphs))]
    test_paragraphs = paragraphs[int(0.7 * len(paragraphs)):]
    train_data = {'data': [{'paragraphs': [p]} for p in train_paragraphs]}
    test_data = {'data': [{'paragraphs': [p]} for p in test_paragraphs]}
    with open(output_folder + '/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(output_folder + '/test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)
    pass


if __name__ == '__main__':
    train_file = 'D:\\FastQA\\data\\Ueno\\all_data_626.pkl'
    folder = 'D:\\FastQA\\data\\Ueno'
    convert_ueno_to_fast_qa(file_path=train_file, output_folder=folder)
