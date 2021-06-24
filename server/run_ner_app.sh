#!/bin/sh
nohup python3 bert_classify_app.py --port 20044 > bert_classify_app.log 2>&1 &
