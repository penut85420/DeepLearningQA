import tensorflow as tf
from bert.modeling import BertConfig, BertModel

def main():
    config = BertConfig.from_json_file('./bert/models/uncased_L-12_H-768_A-12/bert_config.json')
    def model_fn(features, labels, mode, params):
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        model = BertModel(config, True, input_ids, input_mask, segment_ids)
        final_hidden = model.get_sequence_output()
        return final_hidden
    est = tf.estimator.Estimator(model_fn)
    print(est)

if __name__ == "__main__":
    main()
