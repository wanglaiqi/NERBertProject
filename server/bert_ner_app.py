#!/usr/bin/env python
#coding:utf-8

"""
Task: build the python http server
Date:2021.06.24
Author:Laiqi
"""
import os
import sys
import json
import inspect
import argparse
from sanic import Sanic
from sanic.response import text
filename = inspect.getframeinfo(inspect.currentframe()).filename
matrix_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(filename))))
sys.path.insert(0, matrix_dir)
from NERBertProject.server.ner_handler import BertNer
app = Sanic(__name__)
handlerObject = BertNer()

@app.route('/bert/ner',methods=['GET','POST'])
async def input(request):
    sentence = request.args.get('text')
    if sentence is None or sentence.strip() == '""':
        return text("please input questions!!!")
    else:
        result_dict = handlerObject.ner_main_function(sentence)
        # transfer into json format
        result_dict = json.dumps(result_dict, ensure_ascii=False)
        # return the result
        return text(result_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameter about web service')
    parser.add_argument('--port', type=int, default=1111)
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, workers=1, debug=False, access_log=True)
