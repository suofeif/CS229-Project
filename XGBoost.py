import numpy as np
import pandas as pd
from sklearn import svm
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
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
label_number = 0

#We used grid search to pick the best parameters for Gradient Boosting Parameters. We have learning rate=0.1 and max_depth=5 
#as the best parameters for the gradient boosting algorithm. Thus, I commented out codes related to grid search. 
# Grid Search
#params = {'learning_rate': [0.01, 0.05 0.1],
         # 'max_depth': [3,5]
          #}

clf = xgb.XGBClassifier(learning_rate=0.1, max_depth=5)

#cv = GridSearchCV(clf, params, cv = 10, scoring = "f1", n_jobs = -1, verbose = 2)

clf.fit(train_sentence_features, train_labels[:, label_number])

predict = clf.predict(val_sentence_features)
#print("Accuracy rate: ", sum(predict==val_labels[:, label_number])/val_labels.shape[0])

print("Accuracy rate val: ", label_number, sum(predict==val_labels[:, label_number])/val_labels.shape[0])

f1_val=f1_score(val_labels[:, label_number], predict, average='weighted', labels=np.unique(predict))
print("F1 score val:", label_number, f1_val)

predict = clf.predict(test_sentence_features)
# print("Accuracy rate: ", sum(predict==test_labels[:, label_number])/test_labels.shape[0])

print("Accuracy rate test: ", label_number, sum(predict==test_labels[:, label_number])/test_labels.shape[0])

f1_test=f1_score(test_labels[:, label_number], predict, average='weighted', labels=np.unique(predict))
print("F1 score test:", label_number, f1_test)

#print(cv.best_estimator_)
#print(cv.best_params_)


#best_param = {'learning_rate': cv.best_estimator_.learning_rate,
              #'max_depth': cv.best_estimator_.max_depth
              #}


#dtrain = xgb.DMatrix(np.array(train_sentence_features), label= train_labels[:, label_number])
#num_round = 10
#bst = xgb.train(best_param, dtrain, num_round)

#dtest_x = xgb.DMatrix(np.array(val_sentence_features))
#predict = bst.predict(dtest_x)
#total_num = val_labels[:, label_number].shape[0]

#score = 0
#for i, v in enumerate(predict):
    #if np.argmax(v) == val_labels[i, label_number]:
        #score += 1
#print("accuracy is {acc}".format(acc = score/total_num))
