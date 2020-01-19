import sys
from penut.utils import TimeCost

class SQuAD_Dataset:
    def __init__(self, obj):
        self.data = [SQuAD_Data(d) for d in obj['data']]
        self.size = sum([len(d.examples) for d in self.data])

    def iter_examples(self):
        for d in self.data:
            for ex in d:
                yield ex

    def __iter__(self):
        for d in self.data:
            yield d

    def __str__(self):
        rtn = []
        for d in self.data:
            rtn.append(str(d))
        return '\n'.join(rtn)

class SQuAD_Data:
    def __init__(self, obj):
        self.title = obj['title']
        self.examples = [SQuAD_Example(p) for p in obj['paragraphs']]

    def __iter__(self):
        for ex in self.examples:
            yield ex

    def __str__(self):
        rtn = [f'Title: {self.title}']
        for ex in self.examples:
            rtn.append(str(ex))
        return '\n'.join(rtn)

class SQuAD_Example:
    def __init__(self, obj):
        self.context = obj['context']
        self.questions = [SQuAD_Question(qas) for qas in obj['qas']]

    def __iter__(self):
        for q in self.questions:
            yield q

    def __str__(self):
        rtn = [f'Context Length: {len(self.context)}']
        for q in self.questions:
            rtn.append(str(q))
        return '\n'.join(rtn)

class SQuAD_Question:
    qid = None
    question = None
    answer_text = None
    answer_start = None
    is_impossible = None

    def __init__(self, qas):
        self.qid = qas['id']
        self.question = qas['question']
        self.is_impossible = qas['is_impossible']
        if not self.is_impossible:
            self.answer_text = qas['answers'][0]['text']
            self.answer_start = qas['answers'][0]['answer_start']

    def __str__(self):
        rtn = [
            f'QID: {self.qid}',
            f'Question Length: {len(self.question)}',
            f'Is Impossible: {self.is_impossible}',
            f'Answer: {self.answer_text}',
            f'Answer Start: {self.answer_start}',
        ]
        return '\n'.join(rtn)

def demo():
    import penut.io as pio

    d = pio.load('./data/train-v2.0.json')
    with TimeCost('Parsing Time Cost'):
        ds = SQuAD_Dataset(d)

    # qnum = 0
    # for d in ds:
    #     for ex in d:
    #         ctx = ex.context
    #         for q in ex:
    #             question = q.question
    #             answer = q.answer_text
    #             qnum += 1
    #             print(f'{qnum}', end='\r')
    # print(f'Question Numbers: {qnum}')

if __name__ == "__main__":
    # sys.stdout = open('./a.log', 'w', encoding='UTF-8')
    demo()
