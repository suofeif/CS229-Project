import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import f1_score


def load_data_from_csv(file_name, header=0, encoding="utf-8"):

    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df

# define documents
train = load_data_from_csv("sentiment_analysis_trainingset.csv")
val = load_data_from_csv("sentiment_analysis_validationset.csv")
m_val = val.shape[0]
print("finish loading data")

train_doc = train.iloc[:, 1]
val_doc = val.iloc[0:7500, 1]
test_doc = val.iloc[7501:m_val-1:, 1]
print("finish splitting")
print(train_doc.shape)
print(val_doc.shape)
print(test_doc.shape)

# define class labels
train_labels = np.array(train.iloc[:, 2:])
val_labels = np.array(val.iloc[0:7500, 2:])
test_labels = np.array(val.iloc[7501:m_val-1, 2:])
print("finish loading labels")
print(train_labels.shape)
print(val_labels.shape)
print(test_labels.shape)



train_sentence_features = np.load("train_sentence_features.dat")
val_sentence_features = np.load("val_sentence_features.dat")
test_sentence_features = np.load("test_sentence_features.dat")
print(train_sentence_features.shape)
print(val_sentence_features.shape)
print(test_sentence_features.shape)

# This can be from 0 to 19. There are 20 labels.
label_number =19

lin_clf = svm.LinearSVC()
# You can change this with rbf kernel
# rbf_clf = svm.SVC(decision_function_shape='ovo')



lin_clf.fit(train_sentence_features, train_labels[:, label_number])

predict = lin_clf.predict(val_sentence_features)
print("Accuracy rate val: ", label_number, sum(predict==val_labels[:, label_number])/val_labels.shape[0])

f1_val=f1_score(val_labels[:, label_number], predict, average='weighted', labels=np.unique(predict))
print("F1 score val:", label_number, f1_val)

predict = lin_clf.predict(test_sentence_features)
print("Accuracy rate test: ", label_number, sum(predict==test_labels[:, label_number])/test_labels.shape[0])

f1_test=f1_score(test_labels[:, label_number], predict, average='weighted', labels=np.unique(predict))
print("F1 score test:", label_number, f1_test)
