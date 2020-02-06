#!/bin/bash

export ALBERT_MODEL_DIR='./models/albert_base_zh'
export DRCD_DIR='../DRCD'
export OUTPUT_DIR='./outputs/drcd'
export FEATURE_DIR='./features/drcd'

# pip install -r requirements.txt
python run_drcd.py \
  --albert_config_file=$ALBERT_MODEL_DIR/albert_config.json \
  --output_dir=$OUTPUT_DIR \
  --train_file=$DRCD_DIR/DRCD_training.json \
  --predict_file=$DRCD_DIR/DRCD_dev.json \
  --train_feature_file=$FEATURE_DIR/train_feature_file.tf \
  --predict_feature_file=$FEATURE_DIR/predict_feature_file.tf \
  --predict_feature_left_file=$FEATURE_DIR/predict_feature_left_file.tf \
  --init_checkpoint=$ALBERT_MODEL_DIR/model.ckpt-best \
  --vocab_file=$ALBERT_MODEL_DIR/vocab_chinese.txt \
  --do_lower_case \
  --max_seq_length=384 \
  --doc_stride=128 \
  --max_query_length=64 \
  --do_train=true \
  --do_predict=true \
  --train_batch_size=12 \
  --predict_batch_size=8 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --warmup_proportion=.1 \
  --save_checkpoints_steps=5000 \
  --n_best_size=20 \
  --max_answer_length=30
