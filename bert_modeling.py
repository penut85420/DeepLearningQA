import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import tensorflow_hub as hub

from tensorflow import keras
from bert.optimization import AdamWeightDecayOptimizer

def build_model(maxlen):
    bert_path = 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1'

    input_word_ids = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
    input_mask = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
    segment_ids = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32)

    bert_layer = hub.KerasLayer(bert_path, trainable=True)
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    hid = tf.keras.layers.Reshape((maxlen, 768))(sequence_output)
    hid = tf.keras.layers.Flatten()(hid)
    hid = tf.keras.layers.Dense(128)(hid)

    out_begin = tf.keras.layers.Dense(maxlen, name='Begin')(hid)
    out_end = tf.keras.layers.Dense(maxlen, name='End')(hid)

    model = tf.keras.models.Model([input_word_ids, input_mask, segment_ids], [out_begin, out_end])
    opt = AdamWeightDecayOptimizer(learning_rate=3e-5)
    model.compile(opt, 'sparse_categorical_crossentropy', ['acc'])
    model.summary()

    return model

def main():
    maxlen = 128
    bert_path = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'

    x_ids = np.random.randint(0, 5000, (1000, maxlen))
    x_mask = np.random.randint(0, 2, (1000, maxlen))
    x_seg = np.zeros((1000, maxlen))
    y = np.random.randint(0, 2, (1000,))

    input_word_ids = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
    input_mask = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
    segment_ids = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
    bert_layer = hub.KerasLayer(bert_path, trainable=True)
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    out = tf.keras.layers.Dense(1)(pooled_output)

    model = tf.keras.models.Model([input_word_ids, input_mask, segment_ids], out)
    model.compile('adam', 'binary_crossentropy', ['acc'])
    model.summary()

    model.fit([x_ids, x_mask, x_seg], y, batch_size=8, epochs=10, validation_split=0.1)

def main2():
    maxlen = 128

    x_ids = np.random.randint(0, 5000, (1000, maxlen))
    x_mask = np.random.randint(0, 2, (1000, maxlen))
    x_seg = np.zeros((1000, maxlen))
    y = np.random.randint(0, 2, (1000,))

    def initialize_vars(sess):
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        tf.keras.backend.set_session(sess)
    initialize_vars(sess=tf.Session())

    input_word_ids = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
    input_mask = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
    segment_ids = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
    bert_module = hub.Module('https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1', trainable=True)
    bert_inputs = dict(
        input_ids=input_word_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)
    bert_outputs = bert_module(bert_inputs, signature='tokens', as_dict=True)
    pooled_output = bert_outputs['pooled_output']
    out = tf.keras.layers.Dense(1)(pooled_output)

    model = tf.keras.models.Model([input_word_ids, input_mask, segment_ids], out)
    model.compile('adam', 'binary_crossentropy', ['acc'])
    model.summary()

    model.fit([x_ids, x_mask, x_seg], y, batch_size=8, epochs=10, validation_split=0.1)

def main3():
    model = build_model(128)

if __name__ == "__main__":
    main3()
