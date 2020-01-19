import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import penut.io as pio
from squad import SQuAD_Dataset
from penut.utils import TimeCost
from bert_tokenizer import BertTokenizer
from bert_modeling import build_model

def main():
    with TimeCost('Parsing SQuAD'):
        squad_json = pio.load('./data/train-v2.0.json')
        squad_ds = SQuAD_Dataset(squad_json)

    with TimeCost('Loading Bert Tokenizer'):
        bert_path = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
        # bert_path = 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1'
        tk = BertTokenizer(bert_path)

    # with TimeCost('Counting Max Length of Contexts & Questions'):
    #     max_context_length = 0
    #     max_question_length = 0
    #     for i, ex in enumerate(squad_ds.iter_examples()):
    #         print(end=f'{i}/{squad_ds.size}\r')
    #         context = ex.context
    #         context_tokens = tk.tokenize(context)
    #         max_context_length = max(max_context_length, len(context_tokens) + 2)
    #         for q in ex:
    #             question = q.question
    #             question_tokens = tk.tokenize(question)
    #             max_question_length = max(max_question_length, len(question_tokens) + 2)
    #     print(f'Max Context Length: {max_context_length}')
    #     print(f'Max Question Length: {max_question_length}')
    """
    Max Context Length: 853
    Max Question Length: 61
    Counting Max Length: 69.487302
    """

    # max_context_length = 855
    # max_question_length = 63
    # maxlen = max_context_length + max_question_length
    # x, y = [], []
    # with TimeCost('Converting Tokens to IDs'):
    #     for i, ex in enumerate(squad_ds.iter_examples()):
    #         print(end=f'{i}/{squad_ds.size}\r')
    #         for q in ex:
    #             inn = tk.convert_sentence_to_features(q.question, ex.context, maxlen)
    #             x.append(inn)
    #             if q.is_impossible:
    #                 y.append((0, 0))
    #             else:
    #                 answer_tokens = tk.tokenize(q.answer_text)
    #                 y.append((q.answer_start, q.answer_start + len(answer_tokens)))
    # x, y = map(np.array, (x, y))
    # np.save('x', x)
    # np.save('y', y)
    x = np.load('./x.npy')
    y = np.load('./y.npy')
    print(x.shape, y.shape)
    """
    Converting Tokens to IDs: 201.184903
    (130319, 3, 918) (130319, 2)
    """
    model = build_model(maxlen)
    model.fit([x[:, 0], x[:, 1], x[:, 2]], [y[:, 0], y[: 1]], epochs=3, validation_split=0.1)

if __name__ == "__main__":
    main()

"""
===== Result =====
Parsing SQuAD: 0.92902
Loading Bert Tokenizer: 3.68398

"""