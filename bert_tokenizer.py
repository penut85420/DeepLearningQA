import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import tensorflow_hub as hub
from penut.utils import TimeCost
from bert.tokenization import FullTokenizer

class BertTokenizer:
    def __init__(self, bert_path, tokenizer_cls=FullTokenizer, maxlen=512):
        self.maxlen = maxlen
        with tf.compat.v1.Session() as sess:
            bert = hub.Module(bert_path)
            tk_info = bert(signature='tokenization_info', as_dict=True)
            tk_info = [tk_info['vocab_file'], tk_info['do_lower_case']]
            vocab_file, do_lower_case = sess.run(tk_info)
            self.tokenizer = tokenizer_cls(vocab_file, do_lower_case)

    def convert_sentences_to_ids(self, sentences):
        ids = list(map(self.convert_single_sentence_to_ids, sentences))
        return np.array(ids)

    def convert_single_sentence_to_ids(self, sentence):
        tokens = self.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokens += (self.maxlen - len(tokens)) * ['[PAD]']
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_two_sentence_to_ids(self, sent1, sent2, maxlen=None):
        if not maxlen:
            maxlen = self.maxlen
        tokens1 = self.tokenize(sent1)
        tokens2 = self.tokenize(sent2)
        tokens = ['[CLS]'] + tokens1 + ['[SEP]'] + tokens2 + ['[SEP]']
        tokens += (maxlen - len(tokens)) * ['[PAD]']
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def tokenize(self, sent):
        return self.tokenizer.tokenize(sent)
