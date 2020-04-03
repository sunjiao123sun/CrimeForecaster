import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
import pandas as pd
import numpy as np
import datetime
import time
import sklearn
from sklearn.utils.extmath import softmax
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import pickle
from keras.models import Model
from keras.layers import Input, LSTM, Dense, RepeatVector, Permute, Concatenate, Reshape, Softmax, Multiply, TimeDistributed, Flatten
from keras.utils import to_categorical
from keras.engine.topology import Layer, InputSpec
import keras.backend as K
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc, roc_auc_score
import matplotlib.pyplot as plt
# from keras import backend as K
import tensorflow as tf



# def emb(dim_input, dim_emb):
def sigmoid(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            # print("a :", array[i][j])
            array[i][j] = 1/(1 + np.exp(-array[i][j]))
            # print("b :", array[i][j])
    return array

city = 'LA'
# city = 'LA'
month = 12

num_locations = 113
# num_locations = 77
#77 for Chi and 113 for LA
num_categories = 8
num_timeslots = 8

num_embed_loc = 4
num_embed_cat = 16

lstm_dim = 32

# build lstm
event_input = Input(shape=(num_timeslots, num_locations * num_categories), name="event_input")
local_layer = Permute((2,1))(event_input)
local_layer = Reshape((num_locations * num_categories, num_timeslots, 1))(local_layer)
lstm = TimeDistributed(LSTM(lstm_dim))
encoder_outputs = lstm(local_layer) # expected shape(num_locations * num_categories, lstm_dim)
# lstm = LSTM(lstm_dim, return_state=True)
# encoder_outputs, state_h, state_c = lstm(event_input)
state_h = Reshape((num_locations, num_categories, lstm_dim))(encoder_outputs)

# prepare embeddings, here we assume embeddings are learnable
#loc_emb_input = Input(shape=(num_locations, num_locations), name="loc_emb_input")
loc_emb_input = Input(shape=(num_locations, 16), name="loc_emb_input")

cat_emb_input = Input(shape=(num_categories, num_categories), name="cat_emb_input")
loc_emb = Dense(num_embed_loc, name = 'loc_emb_dense')(loc_emb_input)
cat_emb = Dense(num_embed_cat, name = 'cat_emb_dense')(cat_emb_input)
loc_emb = Flatten()(loc_emb)
cat_emb = Flatten()(cat_emb)
# loc_emb_input = Input(shape = (num_locations, num_embed_loc), name = "loc_emb_input")
# cat_emb_input = Input(shape = (num_categories, num_embed_cat), name = "cat_emb_input")
loc_emb = RepeatVector(num_categories)(loc_emb)
loc_emb = Reshape((num_categories, num_locations, num_embed_loc))(loc_emb) # shape(num_categories, num_locations, num_embed_loc)
loc_emb = Permute((2, 1, 3))(loc_emb)
cat_emb = RepeatVector(num_locations)(cat_emb)
cat_emb = Reshape((num_locations, num_categories, num_embed_cat))(cat_emb)
# attention
ita = Concatenate(axis=-1)([state_h, loc_emb, cat_emb]) #shape(num_locations, num_categories, concatsize)
ita = Dense(lstm_dim, activation='tanh', name = 'attention_dense')(ita)  # expected shape(None, num_locations, num_categories, lstm_dim)
alpha = Reshape((num_locations * num_categories, lstm_dim))(ita)
alpha = Softmax(axis=1)(alpha)
alpha = Reshape((num_locations, num_categories, lstm_dim))(alpha)
attention = Multiply()([alpha, state_h])  # expected shape(None, num_locations, num_categories, lstm_dim)
# final output
output = Dense(1, activation="sigmoid", name = 'prediction_dense')(attention)
output = Reshape((num_locations, num_categories))(output)
mist = Model(inputs=[event_input, loc_emb_input, cat_emb_input], outputs=output)
adam = keras.optimizers.Adam(lr=0.3)
def bce_loss(y_true, y_pred):
    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)
    return K.binary_crossentropy(y_true, y_pred)
mist.compile(optimizer='adam', loss=bce_loss)
# mist.compile(optimizer=adam, loss='binary_crossentropy')
# # prepare data
# features = np.load('../Chicago_data_MiST/feature2020-02-28_18:34:57.npy', allow_pickle = True)
# labels = np.load('../Chicago_data_MiST/label2020-02-28_18:34:57.npy', allow_pickle = True)
# features = features.reshape((features.shape[0], num_timeslots, -1))
# locations = list(range(num_locations))
# loc_encoding = to_categorical(locations)
# locs = np.array([loc_encoding for i in range(features.shape[0])])
# categories = list(range(num_categories))
# cat_encoding = to_categorical(categories)
# cats = np.array([cat_encoding for i in range(features.shape[0])])
# features_train, features_test, labels_train, labels_test, locs_train, locs_test, cats_train, cats_test = train_test_split(
#     features, labels, locs, cats, test_size = 0.2)
# print(labels_train.shape)

# CRIME-LA
train = np.load('DCRNN/data/CRIME-%s/%d/train.npz'%(city,month), allow_pickle = True)
test = np.load('DCRNN/data/CRIME-%s/%d/test.npz'%(city,month), allow_pickle=True)
val = np.load('DCRNN/data/CRIME-%s/%d/val.npz'%(city,month),allow_pickle=True)

features_train, features_test, labels_train, labels_test = train["x"], test["x"], train["y"], test["y"]
features_train, features_test, labels_train, labels_test = features_train.reshape((features_train.shape[0],
                                                                                   num_timeslots, -1)), \
                                                           features_test.reshape((features_test.shape[0],
                                                                                  num_timeslots, -1)), \
                                                           np.squeeze(labels_train, axis=1), np.squeeze(labels_test,
                                                                                                        axis = 1)
val_x, val_y = val["x"], val["y"]
val_x, val_y = val_x.reshape((val_x.shape[0],
                              num_timeslots, -1)), \
               np.squeeze(val_y, axis = 1)


# locations = list(range(num_locations))
# loc_encoding = to_categorical(locations)
loc_emb_ge = pd.read_csv('emb_file_la.txt',header=None)
loc_encoding = np.array(loc_emb_ge)
loc_encoding = loc_encoding.astype(np.float32)

categories = list(range(num_categories))
cat_encoding = to_categorical(categories)

locs_train, locs_test, cats_train, cats_test = np.array([loc_encoding for i in range(features_train.shape[0])]), \
                                               np.array([loc_encoding for i in range(features_test.shape[0])]), \
                                               np.array([cat_encoding for i in range(features_train.shape[0])]), \
                                               np.array([cat_encoding for i in range(features_test.shape[0])])

locs_val, cats_val = np.array([loc_encoding for i in range(val_x.shape[0])]), \
                     np.array([cat_encoding for i in range(val_x.shape[0])])

#history = LossHistory()
# train data
history = mist.fit([features_train, locs_train, cats_train], labels_train, validation_data=[[val_x, locs_val,
                                                                                             cats_val], val_y],
                   batch_size=32, epochs=150)
#,callbacks=[history])

y_pred = mist.predict([features_test, locs_test, cats_test])
print(np.shape(labels_test.reshape((labels_test.shape[0],-1))), np.shape(y_pred.reshape((labels_test.shape[0],-1))))



#print("macro-f1:", metrics.f1_score(), y_pred.reshape((labels_test.shape[0],-1)), average = 'macro')
#test_reshape = labels_test.reshape((labels_test.shape[0],-1))
#predict_reshape = y_pred.reshape((labels_test.shape[0],-1))

test_reshape = labels_test.reshape((-1,8))
predict_reshape = y_pred.reshape((-1,8))


#predict_reshape = sigmoid(predict_reshape)
pre_ratio = np.count_nonzero(labels_test) / np.size(labels_test)# the ratio of normal events in the test set
print("The ratio of abnormal events in the test set is:", pre_ratio)
ss = MinMaxScaler(feature_range=(0, 1))
predict_reshape = ss.fit_transform(predict_reshape)
predict_reshape[predict_reshape >= pre_ratio] = 1
predict_reshape[predict_reshape < pre_ratio] = 0

fpr, tpr, thresholds = roc_curve(test_reshape.ravel(), predict_reshape.ravel())
auc_keras = auc(fpr, tpr)

print("auc for MIST: ", auc_keras)
print("macro-f1:", metrics.f1_score(test_reshape, predict_reshape, average = 'macro'))
print("micro-f1:", metrics.f1_score(test_reshape, predict_reshape, average = 'micro'))
print('Final shape of predict_reshape:', predict_reshape.shape)
pickle.dump(test_reshape, open('./result/%s/MIST*/labels_testMIST*%s%d.pkl'%(city,city,month), 'wb'))
pickle.dump(predict_reshape, open('./result/%s/MIST*/predictMIST*%s%d.pkl'%(city,city,month), 'wb'))

# list all data in history
print(history.history.keys())

# summarize history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper right')
plt.savefig('mist_los.png')
print("successfully saved the fig")
