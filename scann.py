from itertools import cycle
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
#import keras
from keras.models import Sequential,load_model
from keras.layers import Dense,Activation,Dropout,Flatten,SimpleRNN,Conv1D,MaxPooling1D,GlobalMaxPool1D
from keras.utils import np_utils
from keras.optimizers import RMSprop,SGD
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score as ARI
import math
from sklearn.metrics import calinski_harabasz_score,roc_curve,auc
from sklearn.model_selection import KFold
from keras.utils.vis_utils import plot_model
import os
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.cluster import KMeans, SpectralClustering
from collections import Counter
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from keras import optimizers

#os.environ["PATH"] += os.pathsep +'C:/Program Files (x86)/Graphviz2.38/bin"'
np.random.seed(123)

# 定义画散点图的函数
def draw_scatter(data,label,length):
    """
    :param n: 点的数量，整数
    :param s:点的大小，整数
    :return: None
    """
    # 加载数据
    # 通过切片获取横坐标x1
    x1 = data[:, 0]
    # 通过切片获取纵坐标R
    y1 = data[:, 1]
    # 创建画图窗口
    fig = plt.figure()
    # 将画图窗口分成1行1列，选择第一块区域作子图
    ax1 = fig.add_subplot(1, 1, 1)
    # 设置标题
    ax1.set_title('Data Visualization')
    # 设置横坐标名称
    ax1.set_xlabel('PC1')
    # 设置纵坐标名称
    ax1.set_ylabel('PC2')
    # 画散点图
    for i in range(length):
        if label[i]==0:
            r=ax1.scatter(x1[i], y1[i],s=20, c='r', marker='.',label='HeLa')
        if label[i]==1:
            b=ax1.scatter(x1[i], y1[i],s=20, c='b', marker='.',label='HAP1')
        if label[i]==2:
            g=ax1.scatter(x1[i], y1[i],s=20, c='g', marker='.',label='GM12878')
        if label[i]==3:
            y=ax1.scatter(x1[i], y1[i],s=20, c='y', marker='.',label='K562')
    plt.legend(handles=[r, b, g, y], loc="upper right")
    # 显示
    plt.show()



# 混淆矩阵定义
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, ('0%', '3%', '5%', '8%', '10%', '12%', '15%', '18%', '20%', '25%'))
    plt.yticks(tick_marks, ('0%', '3%', '5%', '8%', '10%', '12%', '15%', '18%', '20%', '25%'))
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('真实类别')
    plt.xlabel('预测类别')
    plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)
    plt.show()

# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    predictions = model.predict_classes(x_val)
    truelabel = y_val.argmax(axis=-1)  # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel) + 1))

# 卷积网络可视化
def visual(model, data, num_layer=1):
    # data:图像array数据
    # layer:第n层的输出
    layer = keras.backend.function([model.layers[0].input], [model.layers[num_layer].output])
    f1 = layer([data])[0]
    print(f1.shape)
    num = f1.shape[-1]
    print(num)
    plt.figure(figsize=(8, 8))
    for i in range(num):
        plt.subplot(np.ceil(np.sqrt(num)), np.ceil(np.sqrt(num)), i + 1)
        plt.imshow(f1[:, :, i] * 255, cmap='gray')
        plt.axis('off')
    plt.show()

def NMI(A,B):
    # len(A) should be equal to len(B)
    total = len(A)
    idAs = set(A)
    idBs = set(B)
    #Mutual information
    MI = 0
    eps = 1.4e-45
    for idA in idAs:
        for idB in idBs:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # Normalized Mutual information
    Hx = 0
    for idA in idAs:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in idBs:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    result = 2.0*MI/(Hx+Hy)
    return result

def vectorize_sequences(seqs,dim=10000):
    result=np.zeros((len(seqs),dim))
    for i,seq in enumerate(seqs):
        result[i,seq]=1
    return result

def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b
def processing_label(label, truth):
    truth_num=[0,0,0,0]
    for i in truth:
        truth_num[i]=truth_num[i]+1
    label_num=[0,0,0,0]
    for i in label:
        label_num[i]=label_num[i]+1
    index=[0,0,0,0]
    for i in range(4):
        for j in range(4):
            if label_num[i]==truth_num[j]:
                index[i]=j
    for i in range(len(label)):
        for j in range(4):
            if label[i]==index[j]:
                label[i]=j
    return label
#主函数
#data=np.random.random((1000,100))
#data=np.loadtxt('./embedding.txt')#已降维后的单细胞Hi-C矩阵(626,625);
#Label=np.loadtxt('./label.txt',dtype=int,delimiter=' ')
###########dataset
data=np.loadtxt('./embedding_all.txt')#[:,:1000]
copy=data
data=np.random.permutation(data)
Label=np.loadtxt('./label_all.txt',dtype=int,delimiter=' ')
sample_num=data.shape[0]
train_idx=600
valid_idx=2600
label1=[]
for i in range(sample_num):
    for j in range(sample_num):
        if data[i][0]==copy[j][0]:
            label1.append(Label[j])
            continue
labels=[]
for i in label1:
    labels.append([i])
n_labels=np.array(labels)
# #train=vectorize_sequences(data)
train=data
one_hot_labels = keras.utils.to_categorical(n_labels, num_classes=4)
train_data=train[:train_idx]
train_labels=one_hot_labels [:train_idx]

vaild_data=train[train_idx:valid_idx]
valid_labels=one_hot_labels [train_idx:valid_idx]

test_data=train[train_idx:]
test_labels=one_hot_labels [train_idx:]

#真实标签数据可视化图
draw_scatter(test_data,n_labels,test_data.shape[0])


# label_num = []
# for i in range(train_idx):
#     if n_labels[i] == 'HeLa': label_num.append(0)
#     if n_labels[i] == 'HAP1': label_num.append(1)
#     if n_labels[i] == 'GM12878': label_num.append(2)
#     if n_labels[i] == 'K562': label_num.append(3)

###全连接密集网络
learningrate=0.0001
rms = optimizers.RMSprop(lr=learningrate, rho=0.9, epsilon=1e-06)
adam=optimizers.Adam(lr=learningrate)
model1=Sequential()
model1.add(Dense(128,activation='relu',input_dim=train_data.shape[1]))
model1.add(Dropout(0.5))
model1.add(Dense(64,activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(4,activation='softmax'))
model1.compile(optimizer=rms,loss='categorical_crossentropy',metrics=['accuracy'])
model1.summary(0)

history=model1.fit(train_data,train_labels,epochs=500,batch_size=64)

result=model1.evaluate(test_data,test_labels)
print(result)
###交叉验证
# kf=KFold(n_splits=10,shuffle=True,random_state=123)
# k_result=[]
# for i,(train_index,valid_index) in enumerate(kf.split(train)):
#     print("fold:{}, 训练集长度:{}, 验证集长度：{}".format(i, len(train_index), len(valid_index)))
    # x_train,y_train=train[train_index],one_hot_labels[train_index]
    # x_valid, y_valid = train[valid_index], one_hot_labels[valid_index]
    # # print(len(x_train))
    # # print(len(x_valid))
    # ###全连接密集网络
    # model = Sequential()
    # model.add(Dense(128, activation='relu', input_dim=2660))
    # model.add(Dropout(0.5))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4, activation='sigmoid'))
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary(0)
    # history = model.fit(x_train, y_train, epochs=100, batch_size=64)
    # k_result.append(model.evaluate(x_valid, y_valid))

###loss visual
# loss= history1.history['loss']
# val_loss=history1.history['val_loss']
# epochs=range(1,len(loss)+1)
# plt.plot(epochs,loss,'bo',label='Training loss')
# plt.plot(epochs,val_loss,'b',label='Validation loss')
# plt.title('Training loss and  Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend
# plt.show()

###ACC visualize
# acc=history1.history['accuracy']
# val_acc=history1.history['val_accuracy']
# epochs=range(1,len(acc)+1)
# plt.plot(epochs,acc,'bo',label='Training acc')
# plt.plot(epochs,val_acc,'b',label='Validation acc')
# plt.title('Training acc and  Validation acc')
# plt.xlabel('Epochs')
# plt.ylabel('acc')
# plt.legend
# plt.show()


predict1=model1.predict_classes(test_data,batch_size=1)#一维的
draw_scatter(test_data,predict1,test_data.shape[0])

p1=np.array(predict1).flatten()
label1=np.array(label1)
result_nmi1=NMI(label1[train_idx:].flatten(),p1)
print('nmi:',result_nmi1)

result_ari1=ARI(label1[train_idx:].flatten(),p1)
print('ari:',result_ari1)


