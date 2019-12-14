from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import objectives
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import pandas as pd
import keras
from keras.layers import TimeDistributed
from keras.layers import LSTM,RepeatVector
from keras.models import load_model
import json
from keras.models import model_from_json, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras import backend as K
from keras.layers import Merge, Dense
from keras.layers import Reshape
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.utils import to_categorical
import random
import math
import xlrd
import xlsxwriter
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
import functools
from keras.layers import Bidirectional
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from itertools import product
import dill as pickle
from IPython.display import clear_output
from sklearn import mixture
import random
import matplotlib.pyplot as plt
from itertools import groupby
from model import*
#from MYAP1 import*
seed(106)

#######################################################################
#This section loads the data and print the distribution.

d1 = np.array(pd.read_csv('TS1.csv',header=0))
print(d1[:,1])
a = d1[:,1]
a = ([len(list(group)) for key, group in groupby(a)])
print(np.array(a)/d1.shape[0])
d1_input = d1[:,2:]
d1_target =d1[:,1]
d2 = np.array(pd.read_csv('TS2.csv',header=0))
print(d2.shape)
a = d2[:,1]
a = ([len(list(group)) for key, group in groupby(a)])
print(np.array(a)/d2.shape[0])
d2_input = d2[:,2:]
d2_target =d2[:,1]
d3 = np.array(pd.read_csv('TS3.csv',header=0))
print(d3.shape)
a = d3[:,1]
a = ([len(list(group)) for key, group in groupby(a)])
print(np.array(a)/d3.shape[0])
d3_input = d3[:,2:]
d3_target =d3[:,1]
d4 = np.array(pd.read_csv('TS4.csv',header=0))
print(d4.shape)
a = d4[:,1]
a = ([len(list(group)) for key, group in groupby(a)])
print(np.array(a)/d4.shape[0])
d4_input = d4[:,2:]
d4_target =d4[:,1]
d5 = np.array(pd.read_csv('TS5.csv',header=0))
print(d5.shape)
a = d5[:,1]
a = ([len(list(group)) for key, group in groupby(a)])
print(np.array(a)/d5.shape[0])
d5_input = d5[:,2:]
d5_target =d5[:,1]

print('demarcation')

d6 = np.array(pd.read_csv('e-noseforbeefqualitymonitoringdataset.csv',header=0))
print(d6.shape)
a = d6[:,1]
a = ([len(list(group)) for key, group in groupby(a)])
print(np.array(a)/d6.shape[0])
d6_input = d6[:,2:]
d6_target =d6[:,1]


def avgh122():
    list1_input = np.array([d1_input, d2_input ,d3_input ,d4_input, d5_input])
    list1_target = np.array([d1_target,d2_target,d3_target,d4_target,d5_target])
    list2_input = np.array([d6_input])
    list2_target = np.array([d6_target])
    for i in range(1,2):
        Xtrain = list1_input[i];ytrain = list1_target[i]
        clf1, clf2,po,pn,Xtraina, ytraina, Xtrainb, ytrainb =  LSTM_MYAP_TRAIN(Xtrain,ytrain)
        for j in range(0,1):
            Xtest = list2_input[j];ytest = list2_target[j]
            
            re,ra,ada,mia,we,me,hap  = LSTM_MYAP_TEST(clf1, clf2, po, pn, Xtest, ytest,Xtraina, ytraina, Xtrainb, ytrainb)
            acc = re
   
    return (acc)





