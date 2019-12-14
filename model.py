from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import objectives
import numpy as np
import numpy.matlib
import pandas as pd
import keras
from keras.layers import TimeDistributed
from keras.layers import LSTM,RepeatVector
from keras.models import load_model
from keras.models import model_from_json, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras import backend as K
from keras.layers import Merge, Dense
from keras.layers import Reshape
from keras.layers import Input, Embedding, LSTM, Dense, GRU
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
from IPython.display import clear_output
from sklearn import mixture
import random
import matplotlib.pyplot as plt
from itertools import groupby
from sklearn import mixture
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from numpy.random import seed
from keras.callbacks import History
history = History()

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []
    
    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=False)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();

plot_losses = PlotLosses()
#########################################################################################
def LR_train(Xtrain,ytrain):
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(Xtrain, ytrain)
    return clf
#########################################################################################
def cre_newmodel(Xtraina, ytraina, Xtrainb, ytrainb,h1,h2,h3,h4,h5,h6,h7,h8,c1,c2,c3,c4):
    
    la = [c1,c2,c3,c4]
    ha = [h1,h2,h3,h4]
    hb = [h5,h6,h7,h8]
    
    
    
    for n in range(0,4):
        
        if la[n]==1:
            Xtraina = np.concatenate((Xtraina,ha[n] ),  axis = 0)
            ytraina = np.concatenate((ytraina,hb[n] ),  axis = 0)
        else:
            Xtrainb = np.concatenate((Xtrainb, ha[n] ), axis = 0)
            ytrainb = np.concatenate((ytrainb, hb[n] ), axis = 0)



    num_class = 4;num_features = 10;n_epoch = 20;n_batch = 10;look_back = 2
    
    
    
    nb_samples = Xtraina.shape[0] - look_back
    Xtrain2 = np.zeros((nb_samples,look_back,num_features))
    
    y_train_reshaped2 = np.zeros((nb_samples,1,num_class))
    one_hot_labels2 = np.zeros((nb_samples,1,num_class))
    ytra = np.array(pd.get_dummies(np.array(ytraina.astype(int).reshape(-1))))
    
    for i in range(nb_samples):
        y_position = i + look_back
        Xtrain2[i] = Xtraina[i:y_position]
        one_hot_labels2[i] = ytra[y_position,:4]
    
    
    model = Sequential()
    opt = Adam(lr=0.001)
    model.add(LSTM(4,input_shape=(None, num_features), return_sequences=True))
    model.add(TimeDistributed(Dense(num_class,activation = 'tanh')))
    model.add(Activation('softmax'))

    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    filepath="weights-improvement1-{epoch:02d}-{categorical_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='categorical_accuracy', verbose=2, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    clf11 = model.fit(Xtrain2,one_hot_labels2,epochs=n_epoch,batch_size=n_batch,verbose=2)
    clf1 = model
    
    

    hist_df = np.array(clf11.history['loss'])
    cd1 = (hist_df[19])
    
    nb_samples = Xtrainb.shape[0] - look_back
    Xtrain2 = np.zeros((nb_samples,look_back,num_features))
    
    y_train_reshaped2 = np.zeros((nb_samples,1,num_class))
    one_hot_labels2 = np.zeros((nb_samples,1,num_class))
    ytra = np.array(pd.get_dummies(np.array(ytrainb.astype(int).reshape(-1))))
    
    for i in range(nb_samples):
        y_position = i + look_back
        Xtrain2[i] = Xtrainb[i:y_position]
        one_hot_labels2[i] = ytra[y_position,:4]

    model = Sequential()
    opt = Adam(lr=0.001)
    model.add(LSTM(4, input_shape=(None, num_features), return_sequences=True,kernel_initializer='random_uniform'))
    model.add(TimeDistributed(Dense(num_class,activation = 'tanh')))
    model.add(Activation('softmax'))


    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])

    filepath="weights-improvement1-{epoch:02d}-{categorical_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='categorical_accuracy', verbose=2, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]


    n_epoch = 20;
    clf22 = model.fit(Xtrain2,one_hot_labels2, epochs=n_epoch, batch_size=n_batch, verbose=2)
    clf2 = model

   


    hist_df = np.array(clf22.history['loss'])
    cd2 = (hist_df[19])

    return(clf1,clf2,cd1,cd2)






#########################################################################################
def pre(Xtest,ytest):
    Xtraina = Xtest;ytraina = ytest
    num_class = 4
    num_features = 10
    n_epoch = 20
    n_batch = 1
    if len(ytest) < 2:
        look_back = 2
    else:
        look_back = 2
    nb_samples = Xtraina.shape[0] - look_back
    Xtrain2 = np.zeros((nb_samples,look_back,num_features))
    y_train_reshaped2 = np.zeros((nb_samples,1,num_class))
    one_hot_labels2 = np.zeros((nb_samples,1,num_class))
    ytra = np.array(pd.get_dummies(np.array(ytraina.astype(int).reshape(-1))))
    
    for i in range(nb_samples):
        y_position = i + look_back
        Xtrain2[i] = Xtraina[i:y_position]
        one_hot_labels2[i] = ytra[y_position,:num_class]

    return (Xtrain2, one_hot_labels2)



#########################################################################################



def bi(pf):
    look_back = 1
    pf = (pf[0,:])
    k =  (pf[0,:])
    #print(pf[0,:])
    pf = pf[look_back-1,:]
    da = np.argmax(pf)
    pf = pf[da]
    
    return(da,pf,k)



#########################################################################################

def decide(da1,da2,a1,a2,ku,po,pn):
    if da1==ku-1:
        a = 1
    elif da2==ku-1:
        a = 2
    else:
        if po[(ku-1)]>pn[(ku-1)]:
        #if a1>a2:
            a = 1
        else:
            a = 2
    return (a)




#########################################################################################




def te_class(Xtest, ytest):
    Xtrain = Xtest;ytrain = ytest
    no = 4
    
    ku1 = 1;ku2 = 2;ku3 = 3;ku4 = 4;
    
    ytraina = ytrain[ytrain==(ku1)]
    Xtraina = Xtrain[ytrain==(ku1)]
    y1 = ytraina[0:no]
    X1 = Xtraina[0:no,:]

    ytrainb = ytrain[ytrain==(ku2)]
    Xtrainb = Xtrain[ytrain==(ku2)]
    y2 = ytrainb[0:no]
    X2 = Xtrainb[0:no,:]

    ytrainc = ytrain[ytrain==(ku3)]
    Xtrainc = Xtrain[ytrain==(ku3)]
    y3 = ytrainc[0:no]
    X3 = Xtrainc[0:no,:]

    ytraind = ytrain[ytrain==(ku4)]
    Xtraind = Xtrain[ytrain==(ku4)]
    y4 = ytraind[0:no]
    X4 = Xtraind[0:no,:]

    
    return (X1,X2, X3,X4,y1,y2,y3,y4)




#########################################################################################
def train_class(Xtest, ytest, clf1 , clf2, po, pn, l1, l2):

    acc_ = 0
    Xtrain = Xtest;ytrain = ytest
    
    look_back = 2
    ku1 = 1;ku2 = 2;ku3 = 3;ku4 = 4;
    
    ytraina = ytrain[ytrain==(ku1)]
    Xtraina = Xtrain[ytrain==(ku1)]
    ytraina = ytraina[0:4]
    Xtraina = Xtraina[0:4,:]
    
  
    kd = 1
    if len(ytraina) <= 2:
        Xtrainma = np.matlib.repmat(Xtraina[0], 4, 1)
        ytraina = np.matlib.repmat(ytraina[0],  4,  1)
        Xtraine = Xtrainma[0:4,:]
     
        Xtraina, ytraina = pre(Xtrainma,ytraina)
        a1 = clf1.predict(Xtraina);da1, a1,k1 = bi(a1);
        a2 = clf2.predict(Xtraina);da2, a2,k2 = bi(a2);
      
        a = decide(da1,da2,a1,a2,ku1,po,pn);c1 = a
        ytraina = np.matlib.repmat(a, 4, 1)
    
    else:
        Xtrainma = np.matlib.repmat(Xtraina, 1, 1)
       
       
        Xtraine = Xtrainma[0:4,:]
    
        Xtraina,  ytraina = pre(Xtraina,ytraina)
        
        a1 = clf1.predict(Xtraina);da1, a1,k1 = bi(a1);
        a2 = clf2.predict(Xtraina);da2, a2,k2 = bi(a2);
       
        a = decide(da1,da2,a1,a2,ku1,po,pn);c1 = a
        ytraina = np.matlib.repmat(a, 4, 1)
    
  
    acc_ = acc_ +  np.max([k1[0],k2[0]])
    
    ytrainb = ytrain[ytrain==(ku2)]; ytrainb = ytrainb[0:4]
    Xtrainb = Xtrain[ytrain==(ku2)]; Xtrainb = Xtrainb[0:4,:]


    
    kd = 2
    if len(ytrainb) <= 2:
        Xtrainba = np.matlib.repmat(Xtrainb[0], 4, 1)
        ytrainb = np.matlib.repmat(ytrainb[0], 4, 1)
        Xtraine = np.concatenate((Xtraine, Xtrainba[0:4,:] ))
        Xtrainb,  ytrainb = pre(Xtrainba,ytrainb)
        b1 = clf1.predict(Xtrainb);da1, b1,k1 = bi(b1)
        b2 = clf2.predict(Xtrainb);da2, b2,k2 = bi(b2)
       
        b = decide(da1,da2,b1,b2,ku2,po,pn);c2 = b
        ytrainb = np.matlib.repmat(b, 4, 1)
    
    else:
        Xtrainba = np.matlib.repmat(Xtrainb, 1, 1)
        Xtraine = np.concatenate((Xtraine, Xtrainba[0:5,:] ))
        Xtrainb,  ytrainb = pre(Xtrainb,ytrainb)
        b1 = clf1.predict(Xtrainb);da1, b1 ,k1= bi(b1)
        b2 = clf2.predict(Xtrainb);da2, b2, k2 = bi(b2)
       
        b = decide(da1,da2,b1,b2,ku2,po,pn);c2 = b
        ytrainb = np.matlib.repmat(b, 4, 1)
    
   


    acc_ = acc_ +  np.max([k1[1],k2[1]])

    ytrainc = ytrain[ytrain==(ku3)]
    Xtrainc = Xtrain[ytrain==(ku3)]
    ytrainc = ytrainc[0:4]
    Xtrainc = Xtrainc[0:4,:]
    kd = 3
    if len(ytrainc) <= 2:
        Xtrainca = np.matlib.repmat(Xtrainc[0], 4, 1)
        ytrainc = np.matlib.repmat(ytrainc[0], 4, 1)
        Xtraine = np.concatenate((Xtraine,Xtrainca[0:4,:]))
        Xtrainc,  ytrainc = pre(Xtrainca,ytrainc)
        c1 = clf1.predict(Xtrainc);da1, c1, k1 = bi(c1)
        c2 = clf2.predict(Xtrainc);da2, c2, k2 = bi(c2)
        
        c = decide(da1,da2,c1,c2,ku3,po,pn);c3 = c
        ytrainc = np.matlib.repmat(c, 4, 1)

    else:
    
        Xtrainca = np.matlib.repmat(Xtrainc, 1, 1)
        Xtraine = np.concatenate((Xtraine,Xtrainca[0:4,:]))
        Xtrainc,  ytrainc = pre(Xtrainc,ytrainc)
        c1 = clf1.predict(Xtrainc);da1, c1,k1 = bi(c1)
        c2 = clf2.predict(Xtrainc);da2, c2,k2 = bi(c2)
        
        c = decide(da1,da2,c1,c2,ku3,po,pn);c3 = c
        ytrainc = np.matlib.repmat(c, 4, 1)
    
 
    
    #acc_ = np.concatenate((acc_,   ([k1[2],k2[2]]) ), axis = 1)
    acc_ = acc_ +  np.max([k1[2],k2[2]])
    ytraind = ytrain[ytrain==(ku4)]
    Xtraind = Xtrain[ytrain==(ku4)]
    ytraind = ytraind[0:4]
    Xtraind = Xtraind[0:4,:]
    kd = 4
    if len(ytraind) <= 2:
        Xtrainda = np.matlib.repmat(Xtraind[0], 4, 1)
        ytraind = np.matlib.repmat(ytraind[0], 4, 1)
        Xtraine = np.concatenate((Xtraine,Xtrainda[0:4,:]))
        Xtraind,  ytraind = pre(Xtrainda,ytraind)
        d1 = clf1.predict(Xtraind);da1, d1,k1 = bi(d1);
        d2 = clf2.predict(Xtraind);da2, d2,k2 = bi(d2);
       
        d = decide(da1,da2,d1,d2,ku4,po,pn);c4 = d

        ytraind = np.matlib.repmat(d, 4, 1)

        d = decide(da1,da2,d1,d2,ku4,po,pn)

        ytraind = np.matlib.repmat(d, 4, 1)
    else:
        Xtrainda = np.matlib.repmat(Xtraind, 1, 1)
        Xtraine = np.concatenate((Xtraine,Xtrainda[0:4,:]))
        Xtraind,  ytraind = pre(Xtraind,ytraind)
        d1 = clf1.predict(Xtraind);da1, d1,k1 = bi(d1);
        d2 = clf2.predict(Xtraind);da2, d2,k2 = bi(d2);
      
   
        d = decide(da1,da2,d1,d2,ku4,po,pn);c4 = d
        ytraind = np.matlib.repmat(d, 4, 1)


    acc_ = acc_ +  np.max([k1[3],k2[3]])
   
    #Xtraine = np.concatenate((Xtraina,Xtrainb,Xtrainc,Xtraind))
    ytraine = np.concatenate((ytraina,ytrainb,ytrainc,ytraind))
   
    #print('look at me here')
    #print(acc_)
    #model = ctrain(Xtraine, ytraine)
    a = len(np.unique(ytraine))
    b = (np.unique(ytraine))
    if len(np.unique(ytraine)) < 2:
        moda = a
    else:
        moda = LR_train(Xtraine,ytraine)
    return (moda,a,b,c1,c2,c3,c4,acc_)





#########################################################################################



def ctrain(Xtraina,ytraina):
    num_class = 2
    num_features = 10
    n_epoch = 20
    n_batch = 1
    look_back = 2
    

    nb_samples = Xtraina.shape[0] - look_back
    Xtrain2 = np.zeros((nb_samples,look_back,num_features))
    y_train_reshaped2 = np.zeros((nb_samples,1,num_class))
    one_hot_labels2 = np.zeros((nb_samples,1,num_class))
    ytra = np.array(pd.get_dummies(np.array(ytraina.astype(int).reshape(-1))))
    
    for i in range(nb_samples):
        y_position = i + look_back
      
        Xtrain2[i] = Xtraina[i:y_position]
        one_hot_labels2[i] = ytra[y_position,:num_class]

    
    model = Sequential()
    opt = Adam(lr=0.001)
    model.add(LSTM(4,input_shape=(None, num_features), return_sequences=True))
    model.add(TimeDistributed(Dense(num_class,activation = 'tanh')))
    model.add(Activation('softmax'))


    """
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])
    """
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    filepath="weights-improvement1-{epoch:02d}-{categorical_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='categorical_accuracy', verbose=2, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit(Xtrain2,one_hot_labels2, epochs=n_epoch, batch_size=n_batch, verbose=2)
    

    
    return (model)



def add_data(Xtrain,ytrain):
    total = np.hstack((Xtrain , ytrain.reshape(len(ytrain),1)))
    for i in range(1,5):
        data = total[total[:,10]==i]
        rn = random.randint(0,data.shape[0]-1)
        if i == 1:
            every = data[rn:rn+1,:]
        else:
            every = np.concatenate((every, data[rn:rn+1,:]),axis = 0)
    everya = every[:,0:10]
    everyb = every[:,10]
    return (everya, everyb)


def LSTM_MYAP_TRAIN(Xtrain,ytrain):
 
    everya, everyb = add_data(Xtrain,ytrain)
    
    num_class = 2
    num_features = 10
    n_epoch = 20
    n_batch = 10
    look_back = 2
    
    gmm1 = mixture.GaussianMixture(n_components = 2,covariance_type='full').fit(Xtrain)
    nm1 = gmm1.predict(Xtrain)
    nm1 = nm1.reshape(len(nm1),1)
    Xtrain = np.concatenate((Xtrain, nm1),axis = 1);
    Xtrainn = Xtrain ###
    
    ytrainn = Xtrain[:,10 ]
    Xtrain = Xtrain[:,0:Xtrain.shape[1]-1]
    ytraina = ytrain[ytrainn==0]
    ytrainb = ytrain[ytrainn==1]
  
    
################################################################

    
    Xtraina = Xtrainn[Xtrainn[:,10]==0];
    Xtraina = Xtraina[:,0:Xtraina.shape[1]-1]

    Xtrainb = Xtrainn[Xtrainn[:,10]==1];
    Xtrainb = Xtrainb[:,0:Xtrainb.shape[1]-1]
 

    Xtraina = np.concatenate((everya,Xtraina),axis = 0)
    Xtrainb = np.concatenate((everya,Xtrainb),axis = 0)
    ytraina = np.concatenate((everyb,ytraina),axis = 0)
    ytrainb = np.concatenate((everyb,ytrainb),axis = 0)
    
    num_class = 4
    num_features = 10
    n_epoch = 20
    n_batch = 10
    look_back = 2
    
    
    
    nb_samples = Xtraina.shape[0] - look_back
    Xtrain2 = np.zeros((nb_samples,look_back,num_features))
    
    y_train_reshaped2 = np.zeros((nb_samples,1,num_class))
    one_hot_labels2 = np.zeros((nb_samples,1,num_class))
    ytra = np.array(pd.get_dummies(np.array(ytraina.astype(int).reshape(-1))))
    
    for i in range(nb_samples):
        y_position = i + look_back
        Xtrain2[i] = Xtraina[i:y_position]
        one_hot_labels2[i] = ytra[y_position,:4]
    
    
    model = Sequential()
    opt = Adam(lr=0.001)
    model.add(LSTM(4,input_shape=(None, num_features), return_sequences=True))
    model.add(TimeDistributed(Dense(num_class,activation = 'tanh')))
    model.add(Activation('softmax'))
   

    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    filepath="weights-improvement1-{epoch:02d}-{categorical_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='categorical_accuracy', verbose=2, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    cm1 = model.fit(Xtrain2,one_hot_labels2,epochs=n_epoch,batch_size=n_batch,verbose=2)
  

    clf1 = model

    
    nb_samples = Xtrainb.shape[0] - look_back
    Xtrain2 = np.zeros((nb_samples,look_back,num_features))

    y_train_reshaped2 = np.zeros((nb_samples,1,num_class))
    one_hot_labels2 = np.zeros((nb_samples,1,num_class))
    ytra = np.array(pd.get_dummies(np.array(ytrainb.astype(int).reshape(-1))))
    
    for i in range(nb_samples):
        y_position = i + look_back
        Xtrain2[i] = Xtrainb[i:y_position]
        one_hot_labels2[i] = ytra[y_position,:4]
    
    model = Sequential()
    opt = Adam(lr=0.001)
    model.add(LSTM(4, input_shape=(None, num_features), return_sequences=True,kernel_initializer='random_uniform'))
    model.add(TimeDistributed(Dense(num_class,activation = 'tanh')))
    model.add(Activation('softmax'))

    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    filepath="weights-improvement1-{epoch:02d}-{categorical_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='categorical_accuracy', verbose=2, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]



    n_epoch = 20
    cm2 = model.fit(Xtrain2,one_hot_labels2, epochs=n_epoch, batch_size=n_batch, verbose=2)
    clf2 = model
    
    po = ([len(list(group)) for key, group in groupby(np.sort(ytraina))])

    pn = ([len(list(group)) for key, group in groupby(np.sort(ytrainb))])
    

    return (clf1, clf2, po, pn, Xtraina, ytraina, Xtrainb, ytrainb)



#########################################################################################

def LSTM_MYAP_TEST(clf1, clf2, po , pn, Xtest, ytest, Xtraina, ytraina, Xtrainb, ytrainb):
  
    l1 = len(Xtraina)
    l2 = len(Xtrainb)
    moda,a,b,c1,c2,c3,c4,_  = train_class(Xtest,ytest,clf1,clf2, po, pn,l1,l2)
    h1,h2,h3,h4,h5,h6,h7,h8 = te_class(Xtest, ytest)
    clf1,clf2,cd1,cd2 = cre_newmodel(Xtraina, ytraina, Xtrainb, ytrainb, h1, h2, h3, h4, h5, h6, h7, h8, c1, c2, c3, c4)
   
    moda,a,b,c1,c2,c3,c4,acc_ = train_class(Xtest,ytest,clf1,clf2,po,pn,l1,l2)
  
    

    hap = acc_
    
    num_class = 4
    num_features = 10
    look_back = 2
    
    nb_samples = Xtest.shape[0] - look_back
    Xtest2 = np.zeros((nb_samples,look_back,num_features))
    
    y_test_reshaped2 = np.zeros((nb_samples,1, num_class))
    one_hot_labels2 = np.zeros((nb_samples,1, num_class))
    ytes = np.array(pd.get_dummies(np.array(ytest.astype(int).reshape(-1))))
    

    for i in range(nb_samples):
        y_position = i + look_back
        Xtest2[i] = Xtest[i:y_position]
        one_hot_labels2[i] = ytes[y_position,:4]

    
    for i in range(0,Xtest2.shape[0]):
        df = Xtest2[i].reshape(1,2,10)
   
        up = df;up = up.reshape(2,10)
        mama = up[look_back-1]
        pr = clf1.predict(df);pr = (pr[0,:])
        pa = clf2.predict(df);pa = (pa[0,:])
        pr = pr[look_back-1,:]
        pa = pa[look_back-1,:]
            
        ku1 = np.argmax(pr)
        ku2 = np.argmax(pa)
        
        
        if ku1==ku2:
            cb = ku1
        else:
            if a !=2:
                if b==1:
                    cb = ku1
                else:
                    cb = ku2
            else:
                pf = moda.predict(mama.reshape(1,len(mama)))
                #print(pf)
                if pf==1:
                    cb = ku1
                else:
                    cb = ku2

        if i == 0:
            we = np.hstack((pr, pa))
            ada = np.hstack((ku1, ku2))
            pred1 = cb
        else:
            we = np.vstack((we, np.hstack((pr, pa))))
            ada = np.vstack((ada, np.hstack((ku1, ku2))))
            pred1 = np.vstack(( pred1, cb ))
      
      

    ba = len(pred1)
    for j in range(ba):
        if j == 0:
            bc1 = (one_hot_labels2[j,:])
            bc1 = bc1[0,:]
            mali_gd =  np.argmax(bc1)
        else:
            bc1 = (one_hot_labels2[j,:])
            bc1 = bc1[0,:]
            mali_gd = np.vstack(( mali_gd , np.argmax(bc1) ))
    
    mali = pred1
    acc = (mali.reshape(len(mali),1)- mali_gd.reshape(len(mali_gd),1));
    acd = 1- (np.count_nonzero(acc)/ len(mali_gd) )
    mia = np.hstack((po,pn))
    return (acd,mali_gd,ada,mia,we,mali,hap)

#########################################################################################

