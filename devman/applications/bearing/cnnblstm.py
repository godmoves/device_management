import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate # 交叉验证所需的函数
from sklearn.model_selection import KFold,GroupKFold,LeaveOneOut,LeavePOut,ShuffleSplit # 交叉验证所需的子集划分方法
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,TimeDistributed,Dropout,Bidirectional
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.layers.core import Flatten
from sklearn.metrics import mean_squared_error
import math
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# import scipy.stats
# from sklearn.gaussian_process.kernels \
#     import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

for i in range(1,8):
    time_data_path1 = 'D:/data/phm/rs/bearing1_'+str(i)+'rs1hori.csv'
    time_data_path2 = 'D:/data/phm/rs/bearing1_'+str(i)+'rs1vert.csv'
    freq_data_path1 = 'D:/data/phm/rs/bearing1_'+str(i)+'rsfreqhori.csv'
    freq_data_path2 = 'D:/data/phm/rs/bearing1_'+ str(i) + 'rsfreqvert.csv'
    ener_data_path1 = 'D:/data/phm/energy/bearing1_' + str(i) + 'enerhori.csv'
    ener_data_path2='D:/data/phm/energy/bearing1_'+ str(i) + 'enervert.csv'
    df_time1 = pd.read_csv(time_data_path1)
    df_time2 = pd.read_csv(time_data_path2)
    df_freq1 = pd.read_csv(freq_data_path1)
    df_freq2 = pd.read_csv(freq_data_path2)
    df_ener1 = pd.read_csv(ener_data_path1)
    df_ener2 = pd.read_csv(ener_data_path2)

    lenth = len(df_time1.values[:, 0])+1
    p = 0

    data_time1 = df_time1.values[p:, 1:]
    data_time2 = df_time2.values[p:, 1:]
    data_freq1 = df_freq1.values[p:, 1:]
    data_freq2 = df_freq2.values[p:, 1:]
    data_ener1 = df_ener1.values[p+1:, 1:]
    data_ener2 = df_ener2.values[p+1:, 1:]

    # 删除不敏感列
    data_time1 = np.delete(data_time1, [0], axis=1)
    data_time2 = np.delete(data_time2, [0], axis=1)
    data_freq1 = np.delete(data_freq1, [1,3], axis=1)
    data_freq2 = np.delete(data_freq2, [1,3], axis=1)
    data_ener1 = np.delete(data_ener1, [7], axis=1)
    data_ener2 = np.delete(data_ener2, [7], axis=1)

    # 归一化
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    # data_time1 = min_max_scaler.fit_transform(data_time1)
    # data_time2 = min_max_scaler.fit_transform(data_time2)
    data_freq1 = min_max_scaler.fit_transform(data_freq1)
    data_freq2 = min_max_scaler.fit_transform(data_freq2)
    data_ener1 = min_max_scaler.fit_transform(data_ener1)
    data_ener2 = min_max_scaler.fit_transform(data_ener2)


    #data_time1 = preprocessing.scale(data_time1)
    #data_time2 = preprocessing.scale(data_time2)
    #data_freq1 = preprocessing.scale(data_freq1)
    #data_freq2 = preprocessing.scale(data_freq2)

    if(i==1):
        data_31 = np.vstack((data_freq1.T,data_freq2.T,data_ener1.T,data_ener2.T))
        data_31 = data_31.T
        ratio_31=[]
        for j in range(p+2,lenth+1):
            ratio=j / lenth
            ratio_31.append(ratio)
    elif (i==2):
        data_32 = np.vstack((data_freq1.T,data_freq2.T,data_ener1.T,data_ener2.T))
        data_32 = data_32.T
        ratio_32 = []
        for j in range(p+2, lenth + 1):
            ratio =j / lenth
            ratio_32.append(ratio)
    elif (i ==3):
        data_33 = np.vstack((data_freq1.T, data_freq2.T, data_ener1.T, data_ener2.T))
        data_33 = data_33.T
        ratio_33 = []
        for j in range(p+2, lenth + 1):
            ratio = j / lenth
            ratio_33.append(ratio)
    elif (i ==4):
        data_34 = np.vstack((data_freq1.T,data_freq2.T,data_ener1.T,data_ener2.T))
        data_34 = data_34.T
        ratio_34 = []
        for j in range(p+2, lenth + 1):
            ratio = j / lenth
            ratio_34.append(ratio)
    elif (i==5):
        data_35 = np.vstack(( data_freq1.T, data_freq2.T,data_ener1.T,data_ener2.T))
        data_35 = data_35.T
        ratio_35 = []
        for j in range(p+2, lenth + 1):
            ratio = j / lenth
            ratio_35.append(ratio)
    elif (i==6):
        data_36 = np.vstack((data_freq1.T, data_freq2.T,data_ener1.T,data_ener2.T))
        data_36 = data_36.T
        ratio_36 = []
        for j in range(p+2, lenth + 1):
            ratio = j / lenth
            ratio_36.append(ratio)
    elif (i==7):
        data_37 = np.vstack((data_freq1.T, data_freq2.T,data_ener1.T,data_ener2.T))
        data_37 = data_37.T
        ratio_37 = []
        for j in range(p+2, lenth + 1):
            ratio = j / lenth
            ratio_37.append(ratio)

data_3=np.vstack((data_31, data_32))

#pca降维，训练集降维
pca = PCA(n_components=10,copy=True,random_state=8)
pca.fit(data_3)

newdata_3=pca.fit_transform(data_3)

print(pca.explained_variance_ratio_)

newdata_33=pca.transform(data_33)
newdata_34=pca.transform(data_34)
newdata_35=pca.transform(data_35)
newdata_36=pca.transform(data_36)
newdata_37=pca.transform(data_37)
#print(newdata_35.shape)


def get_data (datasetX, datasetY, time_step=20):
    data_x=[]
    data_y=[]
    for i in range(len(datasetX[:,0]) - time_step+1):
        x = datasetX[i:i + time_step, :]
        y = datasetY[i+time_step-1]
        data_x.append(x.tolist())
        data_y.append(y)
    return np.array(data_x),np.array(data_y)


np.random.seed(7)

time_step=80
x_31,y_31=get_data(newdata_3[:len(ratio_31),:], ratio_31, time_step)
x_32,y_32=get_data(newdata_3[len(ratio_31):len(ratio_31)+len(ratio_32), :], ratio_32, time_step)
#x_33,y_33=get_data(newdata_3[len(ratio_31)+len(ratio_32):len(ratio_31)+len(ratio_32)+len(ratio_33), :], ratio_33, time_step)
#x_35,y_35=get_data(newdata_3[len(ratio_31)+len(ratio_32)+len(ratio_33):len(ratio_31)+len(ratio_32)+len(ratio_33)+len(ratio_35),:], ratio_35, time_step)
x_33,y_33=get_data(newdata_33, ratio_33, time_step)
x_34,y_34=get_data(newdata_34, ratio_34, time_step)
x_35,y_35=get_data(newdata_35, ratio_35, time_step)
x_36,y_36=get_data(newdata_36, ratio_36, time_step)
x_37,y_37=get_data(newdata_37, ratio_37, time_step)

# print(x_33.shape)
# print(y_33.shape)

#print(y_35)
#训练集
train_xinitial=np.vstack((x_31,x_32))
print(train_xinitial.shape)
train_yinitial=np.hstack((y_31,y_32))
print(train_yinitial.shape)

ss = ShuffleSplit(n_splits=10, test_size=0.1,random_state=0)
for train, val in ss.split(train_xinitial):
    train_index=train
    val_index=val
    print("随机排列划分：%s %s" % (train_index.shape, val_index.shape))
    break

train_x=[]
train_y=[]
for item in train_index:
    x=train_xinitial[item]
    train_x.append(x)
    y=train_yinitial[item]
    train_y.append(y)
train_x=np.array(train_x)
train_y=np.array(train_y)
print(train_x.shape)

val_x=[]
val_y=[]
for item in val_index:
    x=train_xinitial[item]
    val_x.append(x)
    y=train_yinitial[item]
    val_y.append(y)
val_x=np.array(val_x)
val_y=np.array(val_y)
#print(y_35)
#测试集
test_x=x_37
test_y=y_37
print(test_y.shape)

# create and fit the LSTM network



class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
       # plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.xlabel(loss_type)
        plt.ylabel('loss(mse)')
        plt.legend(loc="upper right")
        plt.show()
        # if loss_type == 'epoch':
        #     # val_acc
        #     plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
        # val_loss
        plt.figure()
        plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        # plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss(mse)')
        plt.legend(loc="upper right")
        plt.show()


#mlp
# model = Sequential()
#
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')

#lstm
# model = Sequential()
#
# model.add(LSTM(45, input_shape=(train_x.shape[1],train_x.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')


#blstm
# model = Sequential()
#
# model.add(Bidirectional(LSTM(65), input_shape=(train_x.shape[1],train_x.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')

#
#
#cnnblstm
train_xinitial=train_xinitial.reshape(train_xinitial.shape[0],2,int(train_xinitial.shape[1]/2),train_xinitial.shape[2])
train_x=train_x.reshape(train_x.shape[0],2,int(train_x.shape[1]/2),train_x.shape[2])
test_x=test_x.reshape(test_x.shape[0],2,int(test_x.shape[1]/2),test_x.shape[2])
val_x=val_x.reshape(val_x.shape[0],2,int(val_x.shape[1]/2),val_x.shape[2])
model = Sequential()
#
model.add(TimeDistributed(Conv1D(filters=128, kernel_size=1, activation='tanh'), input_shape=(None, train_x.shape[2],train_x.shape[3])))
model.add(TimeDistributed(MaxPooling1D(pool_size=4)))
# # #print(model.summary())
# model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='tanh'), input_shape=(None, train_x.shape[2],train_x.shape[3])))
# model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
# # #print(model.summary())
# model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='tanh'), input_shape=(None, train_x.shape[2],train_x.shape[3])))
# model.add(TimeDistributed(MaxPooling1D(pool_size=4)))
# model.add(TimeDistributed(Conv1D(filters=32, kernel_size=2, activation='tanh'), input_shape=(None, train_x.shape[2],train_x.shape[3])))
# model.add(TimeDistributed(MaxPooling1D(pool_size=4)))
#
model.add(TimeDistributed(Flatten()))
# print(model.summary())
model.add(Bidirectional(LSTM(60,return_sequences=True)))
model.add(Bidirectional(LSTM(48,return_sequences=True)))
model.add(Bidirectional(LSTM(48)))
model.add(Dropout(0.5))
model.add(Dense(1))
print(model.summary())
model.compile(loss='mean_squared_error', optimizer='adam')


history = LossHistory()
model.fit(train_x, train_y, validation_data=(val_x,val_y),epochs=100, batch_size=100, verbose=2,callbacks=[history])
#model.fit(train_x, train_y,epochs=70, batch_size=100)



# model.layers  ,layer.weights

# names = [weight.name for layer in model.layers for weight in layer.weights]
# weights = model.get_weights()
# for name, weight in zip(names, weights):
#     print(name, weight.shape)

# make predictions

trainPredict = model.predict(train_xinitial)
testPredict = model.predict(test_x)

print(trainPredict[:,0])


trainScore = math.sqrt(mean_squared_error(train_yinitial, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))

print(testPredict[:,0])


# trainPred={}
# for i in range(len(train_y)):
#     trainPred[train_y[i]]=trainPredict[i,0]
# print(trainPred)
# print(sorted(trainPred))
# sorted_trainy=sorted(trainPred)
# sorted_y = sorted(trainPred.items(),key = lambda trainPred: trainPred[0])
# sorted_trainPredict=np.array(sorted_y)[:,1]
#
# print(np.array(sorted_y).shape)

plt.plot(train_yinitial,label=u"true")
plt.plot(trainPredict[:,0],"r",label=u"pred")
plt.show()


plt.plot(testPredict[:,0],"r",label=u"pred")
plt.show()

history.loss_plot('epoch')

trainPredict_11=trainPredict[:len(y_31),0]
trainPredict_12=trainPredict[len(y_31):,0]
Predict_13=testPredict[:,0]
df_predict=pd.DataFrame(np.array(Predict_13).T)
df_train11=pd.DataFrame(np.array(trainPredict_11).T)
df_train12=pd.DataFrame(np.array(trainPredict_12).T)
#df_train11.to_csv('D:/data/phm/1_1.csv')
#df_train12.to_csv('D:/data/phm/1_2.csv')
df_predict.to_csv('D:/data/phm/1_7cnnblstm.csv')

