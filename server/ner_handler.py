#!/usr/bin/env python
#coding:utf-8
"""
task: use the bert model for train classify model
date: //2021.06.24
author: laiqi
"""
import os
import sys
import inspect
import numpy as np
import tensorflow as tf
filename = inspect.getframeinfo(inspect.currentframe()).filename
matrix_dir = os.path.dirname(os.path.dirname(os.path.abspath(filename)))
sys.path.insert(0, matrix_dir)
from datetime import datetime
from NERBertProject.bert import modeling
from NERBertProject.bert import tokenization
from NERBertProject.tool.tools import load_yaml_file
from NERBertProject.data.prepare_data import InputExample
from NERBertProject.data.prepare_data import NerProcessor
from NERBertProject.data.prepare_data import InputFeatures
from NERBertProject.data.prepare_data import PaddingInputExample
os.environ["CUDA_VISIBLE_DEVICES"]='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#set use the memory
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

class BertNer(object):

    def __init__(self):
        self.debug = False
        # get the config file
        self.config_dict = load_yaml_file(os.path.join(matrix_dir, 'config/config.yaml'))
        # get the bert base model path
        self.bert_model_path = self.config_dict["init_model_path"]
        self.bert_config_file = self.bert_model_path + "bert_config.json"
        # get the vocab file
        self.vocab_file = self.bert_model_path + "vocab.txt"
        # get the output path
        self.init_checkpoint = self.config_dict["model_path"]
        # the max sequence length
        self.max_seq_length = 128
        # 只设为1，目前每次只预测一句话的标签 None according the number of input
        self.batch_size = None
        self.do_lower_case = True
        self.is_training = False
        # 使用GPU和CPU时，此参数为False，使用TPU时此参数为True
        self.use_one_hot_embeddings = False
        # 加载bert模型配置参数
        self.bert_config = modeling.BertConfig.from_json_file(os.path.join(matrix_dir, self.bert_config_file))
        # define object about data processor
        self.processor = NerProcessor()
        # deal with label
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        self.label2id = {label:i for (i, label) in enumerate(self.label_list)}
        self.id2label = {i:label for (i, label) in enumerate(self.label_list)}
        # define tokenizer
        self.tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(matrix_dir, self.vocab_file), do_lower_case=self.do_lower_case)
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            self.input_ids_p = tf.placeholder(tf.int32, [None, self.max_seq_length], name="input_ids")
            self.input_mask_p = tf.placeholder(tf.int32, [None, self.max_seq_length], name="input_mask")
            self.segment_ids_p = tf.placeholder(tf.int32, [None, self.max_seq_length], name="segment_ids")
            self.label_ids_p = tf.placeholder(tf.int32, [None, self.max_seq_length], name="label_ids")
            (self.total_loss, self.per_example_loss, self.logits, self.pedict) = self.create_model(
                self.bert_config, self.is_training, self.input_ids_p, self.input_mask_p, self.segment_ids_p,
                self.label_ids_p, self.num_labels, self.use_one_hot_embeddings)
            saver = tf.compat.v1.train.Saver()
            # saver.restore(sess, tf.train.latest_checkpoint(self.init_checkpoint))
            model_abs_dir = self.get_latest_checkpoint(os.path.join(matrix_dir, self.init_checkpoint))
            # 加载ner模型
            saver.restore(sess, model_abs_dir)

    # step1.2:load error corpus data into memory
    def get_latest_checkpoint(self, model_dir):
        checkpoint_dir = os.path.join(model_dir, "checkpoint")
        with open(checkpoint_dir) as file:
            for line in file:
                line = line.strip()
                if "model_checkpoint_path" in line:
                    model_checkpoint_path = line.split(": ")[1]
                    model_checkpoint_path = model_checkpoint_path.strip().strip('"')
                    newest_model_name = os.path.basename(model_checkpoint_path)
                    abs_model_dir = os.path.join(model_dir, newest_model_name)
                    return abs_model_dir
        raise Exception("no checkpoint was found.....")

    # step0: create model creates a ner model.
    def create_model(self, bert_config, is_training, input_ids, input_mask, segment_ids,
                     labels, num_labels, use_one_hot_embeddings):
        """Creates a ner model."""
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        # In the demo, we are doing a simple classification task on the entire
        # segment.          单分类
        #
        # If you want to use the token-level output, use model.get_sequence_output()
        # instead.   例如 NER

        # output_layer = model.get_pooled_output()

        output_layer = model.get_sequence_output()

        hidden_size = output_layer.shape[-1].value

        output_weight = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer()
        )
        # loss 和 predict 需要自己定义
        with tf.variable_scope("loss"):
            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            output_layer = tf.reshape(output_layer, [-1, hidden_size])
            logits = tf.matmul(output_layer, output_weight, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            # 10 类
            logits = tf.reshape(logits, [-1, self.max_seq_length, 10])
            # mask = tf.cast(input_mask,tf.float32)
            # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
            # return (loss, logits, predict)
            ##########################################################################
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_sum(per_example_loss)
            probabilities = tf.nn.softmax(logits, axis=-1)
            predict = tf.argmax(probabilities, axis=-1)

            return (loss, per_example_loss, logits, predict)

    # step1: convert single example
    def convert_single_example(self, example, label_list, max_seq_length, tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        if isinstance(example, PaddingInputExample):
            return InputFeatures(
                input_ids=[0] * max_seq_length,
                input_mask=[0] * max_seq_length,
                segment_ids=[0] * max_seq_length,
                label_id=0,
                is_real_example=False)

        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i
        tokens_a = tokenizer.tokenize(example.text_a)

        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        labellist = example.label.split(' ')
        if len(labellist) > (max_seq_length - 2):
            labellist = labellist[0:(max_seq_length - 2)]

        label_id = []
        label_id.append(label_map["[CLS]"])
        for i in labellist:
            if i in label_map.keys():
                label_id.append(label_map[i])
            else:
                label_id.append(label_map['PAD'])

        label_id.append(label_map["[SEP]"])
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_id.append(label_map['PAD'])
        if len(label_id) != max_seq_length:
            print(len(input_ids), len(label_id))
            print(example.text_a)
            print(tokens)
            print(input_ids, label_id)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_id) == max_seq_length

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            is_real_example=True)
        # return feature
        return feature

    # step:1.1: modify the length of sentence
    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        """
        这是一个简单的启发式方法，它总是会一次截断一个令牌的较长序列。 这比从每个令牌中截取相等百分比的令牌更有意义，因为如果一个序列非常短，
        那么被截断的每个令牌可能包含比更长序列更多的信息。
        """
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    # step1.2:sort out result
    def get_final_result(self, sentence_split, sentence_label):
        # define the result_dict
        result_dict = dict()
        per, loc, org = '', '', ''
        for word, tag in zip(sentence_split, sentence_label):
            if tag in ('B-PER', 'I-PER'):
                per += ' ' + word if (tag == 'B-PER') else word
            if tag in ('B-ORG', 'I-ORG'):
                org += ' ' + word if (tag == 'B-ORG') else word
            if tag in ('B-LOC', 'I-LOC'):
                loc += ' ' + word if (tag == 'B-LOC') else word
        if per:
            result_dict["person"] = per.lstrip().split(" ")
        if loc:
            result_dict["location"] = loc.lstrip().split(" ")
        if org:
            result_dict["organzation"] = org.lstrip().split(" ")
        # return the result
        return result_dict

    # define main_function
    def ner_main_function(self, question):
        # get the text_a
        text_a = " ".join([word for word in question])
        # build the predict_example
        predict_example = InputExample(guid="predict-0", text_a=text_a, text_b=None, label=" ".join(len(text_a.split(" "))*["O"]))
        # convert single example
        feature = self.convert_single_example(predict_example, self.label2id, self.max_seq_length, self.tokenizer)

        # 每次处理len(sentences)的批次大小，即batch_size为len(sentences)
        input_ids = np.reshape([feature.input_ids], (1, self.max_seq_length))
        input_mask = np.reshape([feature.input_mask], (1, self.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], (1, self.max_seq_length))

        with self.graph.as_default():
            feed_dict = {self.input_ids_p: input_ids, self.input_mask_p: input_mask, self.segment_ids_p: segment_ids}
            predict_result = sess.run([self.pedict], feed_dict)
            # get the predict id
            predict_id = predict_result[0][0]
            # get the predict tag
            predict_tag = [self.id2label[id] for id in predict_id if id != 0]
            # delete the token [CLS] and [SEP]
            predict_tag = [val for val in predict_tag if val!="[CLS]" and val!="[SEP]"]
            # get the final result
            result_dict = self.get_final_result(text_a.split(), predict_tag)
        # return the result
        return result_dict

if __name__ == "__main__":
    # the start_time
    start_time = datetime.now()
    # define object
    handlerObject = BertNer()
    # test question
    questions = ["北京,美国的华莱士，我和他谈笑风生。", "张三和李四去天津玩耍"]
    label = ["B-LOC", "I-LOC", "O", "B-PER", "B-PER", "B-PER", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
    # call the getIntent function
    result_dict = handlerObject.ner_main_function(questions[1])
    print(result_dict)
    print("=**="*10)
    # the end_time
    end_time = datetime.now()
    use_time = end_time - start_time
    print("run time is:%ss||time:%s" % (use_time.seconds, use_time))
