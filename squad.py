import sys
from penut.utils import TimeCost

class SQuAD_Dataset:
    data = []

    def __str__(self):
        rtn = []
        for d in self.data:
            rtn.append(str(d))
        return '\n'.join(rtn)

    @classmethod
    def parsing(cls, obj):
        ds = SQuAD_Dataset()
        for d in obj['data']:
            dd = SQuAD_Data.parsing_json(d)
            ds.data.append(dd)
        return ds

class SQuAD_Data:
    examples = []
    title = []

    def __str__(self):
        rtn = [f'Title: {self.title}']
        for ex in self.examples:
            rtn.append(str(ex))
        return '\n'.join(rtn)

    @classmethod
    def parsing_json(cls, obj):
        d = SQuAD_Data()
        d.title = obj['title']
        for p in obj['paragraphs']:
            ex = SQuAD_Example.parsing(p)
            d.examples.append(ex)
        return d

class SQuAD_Example:
    context = None
    questions = []

    def __str__(self):
        rtn = [f'Context Length: {len(self.context)}']
        for q in self.questions:
            rtn.append(str(q))
        return '\n'.join(rtn)

    @classmethod
    def parsing(cls, obj):
        ex = SQuAD_Example()
        ex.context = obj['context']
        for qas in obj['qas']:
            q = SQuAD_Question()
            q.qid = qas['id']
            q.question = qas['question']
            q.is_impossible = qas['is_impossible']
            if not q.is_impossible:
                q.answer_text = qas['answers'][0]['text']
                q.answer_start = qas['answers'][0]['answer_start']
            ex.questions.append(q)
        return ex

class SQuAD_Question:
    qid = None
    question = None
    answer_text = None
    answer_start = None
    is_impossible = None

    def __str__(self):
        rtn = [
            f'QID: {self.qid}',
            f'Question Length: {len(self.question)}',
            f'Is Impossible: {self.is_impossible}',
            f'Answer: {self.answer_text}',
            f'Answer Start: {self.answer_start}',
        ]
        return '\n'.join(rtn)

def main():
    import penut.io as pio

    d = pio.load('./data/train-v2.0.json')
    with TimeCost('Parsing Time Cost'):
        ds = SQuAD_Dataset.parsing(d)

    for d in ds.data:
        for ex in d.examples:
            for q in ex:
                pass

if __name__ == "__main__":
    sys.stdout = open('./a.log', 'w', encoding='UTF-8')
    main()
