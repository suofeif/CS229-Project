
Fine-Grained Sentiment Analysis of Restaurant Customer Reviews in Chinese Language
=========================================

* The baseline model (main_train.py, main_predict.py, model.py, data_process.py, config.py) is provided by AI Challenger official. We trained the SVC with rbf kernel. Here is the link to official site: https://github.com/AIChallenger/AI_Challenger_2018/tree/master/Baselines/sentiment_analysis2018_baseline.
* All the other code files are produced by the teammates.

Simple instructions:
---
* Datasets are available at: https://challenger.ai/competition/fsauor2018
* prc_save_Word2Vec.py load and output embedding matrix and indices files for train, val, and test data.
* pre_processing_ref.py provides utility functions for all the other files.
* keras_rnn.py is for LSTM model using w2v representations.
* Continue_w2v.py is for further tuning and training on a subset of elements with lower performance in the output of keras_rnn.py.
* GradientBoosting.py for XGBoost classification model.
* SVM.py for linear SVC.
* crate_sentence_features.py outputs sentence features for XGBoost and SVC models.
