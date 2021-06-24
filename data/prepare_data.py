#!/usr/bin/env python
#coding:utf-8
"""
Task: deal with dataset
Date: 2021.06.23
Author:Laiqi
"""
import os
import sys
import inspect
filename = inspect.getframeinfo(inspect.currentframe()).filename
matrix_dir = os.path.dirname(os.path.dirname(os.path.abspath(filename)))
sys.path.insert(0, matrix_dir)
from datetime import datetime
from NERBertProject.bert import tokenization
from NERBertProject.tool.tools import load_yaml_file

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def read_data(self, input_file):
        """Reads a BIO data."""
        with open(input_file) as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                # if len(contends) == 0 and words[-1] == '。':
                if len(contends) == 0:
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            # return lines
            return lines

class NerProcessor(DataProcessor):

    def __init__(self):
        # get the config file
        self.config_dict = load_yaml_file(os.path.join(matrix_dir, 'config/config.yaml'))
        # get the data path
        self.data_path = os.path.join(matrix_dir, self.config_dict["data_path"])

    def get_train_examples(self, data_dir):
        return self.create_example(
            self.read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self.create_example(
            self.read_data(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self,data_dir):
        return self.create_example(
            self.read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        # prevent potential bug for chinese text mixed with english text
        # return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]","[SEP]"]
        return ["PAD", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC","O","[CLS]","[SEP]"]

    def create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
        # return the result
        return examples

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

if __name__ == "__main__":
    # the start_time
    start_time = datetime.now()
    # get the data_dir
    data_dir = os.path.join(matrix_dir, "resources/dataset/")
    # define object
    dataObject = NerProcessor()
    # get label
    label_list = dataObject.get_labels()
    # # get train examples
    # train_examples = dataObject.get_train_examples(data_dir)
    # get dev examples
    dev_examples = dataObject.get_dev_examples(data_dir)
    # get test examples
    test_examples = dataObject.get_test_examples(data_dir)
    # the end_time
    end_time = datetime.now()
    use_time = end_time - start_time
    print("run time is:%ss||time:%s" % (use_time.seconds, use_time))