#! /bin/bash
# step0: define the base path
export data_path=../resources/dataset/
export pretrained_model_path=../resources/initModel/chinese_wwm_ext_L-12_H-768_A-12
export output_model_path=../resources/model/ner_model_v2

# step1: train the classify model
python3.6 ./run_bert_lstm_crf.py \
--data_dir=$data_path \
--task_name=ner \
--vocab_file=$pretrained_model_path/vocab.txt \
--init_checkpoint=$pretrained_model_path/bert_model.ckpt \
--bert_config_file=$pretrained_model_path/bert_config.json \
--max_seq_length=128 \
--train_batch_size=12 \
--learning_rate=2e-5 \
--num_train_epochs=1.0 \
--do_train=True \
--do_eval=True \
--do_predict=True \
--output_dir=$output_model_path
