import numpy as np
import utils
import argparse
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib.pyplot as plt
from matplotlib import colors

#分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB#categoricalNB离散化的朴素贝叶斯分类器
from sklearn.svm import SVC
#评价指标、准确率
from sklearn.metrics import confusion_matrix

#用程序的参数来确定目前使用的数据集和训练方法
parser=argparse.ArgumentParser('Conventional Classifiers')

parser.add_argument('dataset',choices=['Indian','Pavia','Houston'],default='Indian')
parser.add_argument('mode',choices=['KNN','NB','SVM','RF'],default='KNN')

args=parser.parse_args()

data,data_TE,data_TR,data_label=utils.loaddata(args.dataset)

# 标准化数据

#创建一个同样大小的矩阵用于保存数据
input_normalize = np.zeros(data.shape)
#data。shape【2】指的是对于每一个波段来做标准化，对于每一个波段，找最大最小值，再归一化
for i in range(data.shape[2]):
    input_max = np.max(data[:, :, i])
    input_min = np.min(data[:, :, i])
    input_normalize[:, :, i] = (data[:, :, i] - input_min) / (input_max - input_min)

total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true=utils.chooose_train_and_test_point(data_TR,data_TE,data_label,16)
#number_train==[50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 15, 15, 15]
#number_test==[1384, 784, 184, 447, 697, 439, 918, 2418, 564, 162, 1244, 330, 45, 39, 11, 5]
#number_true==[10659, 1434, 834, 234, 497, 747, 489, 968, 2468, 614, 212, 1294, 380, 95, 54, 26, 20]
#number_true多一个维数是背景像素，10659个

#sum(number_train)==695，恰好等于total_pos_train里元素的数量

width,height,band=input_normalize.shape

#将pos和label转换为分类器所需要的形式
#X_train x个像素*200个波段
#y_train 1*一个label
#一一对应

#input_normalize 145*145*200,其中所有的数字都被限制在（0，1）之间
x_train,x_test,x_true=utils.toStandardformX(input_normalize, band, total_pos_train, total_pos_test, total_pos_true,number_train, number_test, number_true)
#x_train=659*200,659个坐标，200个波段

y_train,y_test,y_true=utils.toStandardformY(data_label, band, total_pos_train, total_pos_test, total_pos_true,number_train, number_test, number_true)


global pre,pree
if args.mode=='KNN':
    #使用的是KNN
    #k值等于10
    knn=KNeighborsClassifier(n_neighbors=10)
    knn.fit(x_train,y_train)
    pre= knn.predict(x_test)
    pree=knn.predict(x_true)
elif args.mode=='RF':
    #使用随机森林
    #n_estimate值设置为200，即代表着200棵决策树
    RF=RandomForestClassifier(n_estimators=200)
    RF.fit(x_train,y_train)
    pre=RF.predict(x_test)
    pree=RF.predict(x_true)
elif args.mode=='NB':
    #使用朴素贝叶斯算法来预测
    #***************************有使用的bug，使用不了
    NB=CategoricalNB(min_categories=16)
    NB.fit(x_train, y_train)
    pre = NB.predict(x_test)
    pree=NB.predict(x_true)
elif args.mode=='SVM':
    #最好使用五折交叉认证来确定最好的c和gamma
    #gamma的取值范围
    SVM=SVC(kernel='rbf',C=10000,gamma=0.01)
    SVM.fit(x_train, y_train)
    pre = SVM.predict(x_test)
    pree=SVM.predict(x_true)

matrix = confusion_matrix(y_test, pre)
#混淆矩阵的每一行表示某一类真实值，全部加起来等于某一类的总数
utils.cal_class_results(matrix,y_train)

classification_result=np.zeros(shape=(width,height),dtype=float)



for i in range(sum(number_true)):
    classification_result[total_pos_true[i][0],total_pos_true[i][1]]+=pree[i]
savemat('RF-Pavia.mat', mdict={"result":classification_result})



OA, AA_mean, Kappa, AA = utils.cal_results(matrix)
print()
print("Overall metrics:")
print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA_mean, Kappa))