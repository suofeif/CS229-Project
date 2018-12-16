import numpy as np


train_indices = np.load("train_indices.dat")
val_indices = np.load("val_indices.dat")
test_indices = np.load("test_indices.dat")

embedding_matrix = np.load("embedding_matrix.dat")


nmb_of_train_ex, train_sentence_len = train_indices.shape
nmb_of_val_ex, val_sentence_len = val_indices.shape
nmb_of_test_ex, test_sentence_len = test_indices.shape

_, word_dimension = embedding_matrix.shape

train_sentence_features = np.zeros((nmb_of_train_ex, word_dimension))
val_sentence_features = np.zeros((nmb_of_val_ex, word_dimension))
test_sentence_features = np.zeros((nmb_of_test_ex, word_dimension))

for i in range(nmb_of_train_ex):
    for j in range(train_sentence_len):
        train_sentence_features[i, :] += embedding_matrix[int(train_indices[i, j]), :]


for i in range(nmb_of_val_ex):
    for j in range(val_sentence_len):
        val_sentence_features[i, :] += embedding_matrix[int(val_indices[i, j]), :]


for i in range(nmb_of_test_ex):
    for j in range(test_sentence_len):
        test_sentence_features[i, :] += embedding_matrix[int(test_indices[i, j]), :]


train_sentence_features.dump("train_sentence_features.dat")
val_sentence_features.dump("val_sentence_features.dat")
test_sentence_features.dump("test_sentence_features.dat")
