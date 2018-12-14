
Fine-Grained Sentiment Analysis of Restaurant Customer Reviews in Chinese Language
=========================================

* The baseline model (main_train.py, main_predict.py, model.py, data_process.py, config.py) is provided by AI Challenger official. We trained the SVC with rbf kernel. Here is the link to official site: https://github.com/AIChallenger/AI_Challenger_2018/tree/master/Baselines/sentiment_analysis2018_baseline.

* This project is for both cs229 and cs230. The codes special for cs229 and cs230 are respectively listed below.

Simple instructions:
---
* Datasets are available at: https://challenger.ai/competition/fsauor2018
* prc_save_Word2Vec.py load and output embedding matrix and indices files for train, val, and test data.
* pre_processing_ref.py provides utility functions for all the other files.
* keras_rnn.py is for LSTM model using w2v representations.
* Continue_w2v.py is for further tuning and training on a subset of elements with lower performance in the output of keras_rnn.py.



Specially for CS229:
---
*
*
*

Specially for CS230:
---
* 
*
*
