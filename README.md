# NERProject
基于预训练语言模型BERT的中文命名实体识别样例

# 1：项目的目录结构如下
── bert  #bert的代码块
│   ├── modeling.py
│   ├── optimization.py
│   ├── tf_metrics.py
│   └── tokenization.py
├── config   #配置文件
│   ├── config.yaml
├── data  #数据处理模块
│   └── prepare_data.py
├── LICENSE
├── model  #模型训练模块
│   ├── lstm_crf_layer.py
│   ├── run_bert_lstm_crf.py  #基于bert+lstm+crf的模型
│   ├── run_bert_lstm_crf.sh
│   ├── run_bert_ner.py  #基于bert的模型
│   └── run_bert_ner.sh
├── README.md
├── resources
│   ├── dataset  #开源的数据集
│   │   ├── dev.txt
│   │   ├── test.txt
│   │   └── train.txt
│   ├── initModel  #开源的预训练模型
│   │   └── chinese_wwm_ext_L-12_H-768_A-12
│   │       ├── bert_config.json
│   │       ├── bert_model.ckpt.data-00000-of-00001
│   │       ├── bert_model.ckpt.index
│   │       ├── bert_model.ckpt.meta
│   │       └── vocab.txt
│   └── model   #训练完模型的保存路径
│       ├── ner_model_v1
│       │   ├── eval_results.txt
│       │   └── test_prediction.txt
│       └── ner_model_v2
│           ├── eval_results.txt
│           ├── test_predictionv2.txt
├── server  #web服务部署模块
│   ├── bert_ner_app.py
│   ├── ner_handler.py  #推理模块
│   └── run_ner_app.sh  #启动web服务
└── tool  #工具类
    └── tools.py
