import argparse
import utils
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser=argparse.ArgumentParser("1DNN")
parser.add_argument('dataset',choices=['Indian','Houston','Pavia'],default='Indian')
parser.add_argument('epoch', type=int, default=500, help='epoch number')
args=parser.parse_args()

data,data_TE,data_TR,data_label=utils.loaddata(args.dataset)

input_normalize=utils.normalize(data)

total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = utils.chooose_train_and_test_point(
    data_TR, data_TE, data_label, 16)
width,height,band=input_normalize.shape

# input_normalize 145*145*200,其中所有的数字都被限制在（0，1）之间
x_train, x_test, x_true = utils.toStandardformX(input_normalize, band, total_pos_train, total_pos_test,
                                          total_pos_true, number_train, number_test, number_true)
# x_train=659*200,659个坐标，200个波段

y_train, y_test, y_true = utils.toStandardformY(data_label, band, total_pos_train, total_pos_test, total_pos_true,
                                          number_train, number_test, number_true)

# 将numpy的ndarray类型转换为torch的tensor类型，再用tensordataset读取
X_train = torch.from_numpy(x_train).type(torch.FloatTensor)
Y_train = torch.from_numpy(y_train).type(torch.LongTensor)
Label_train = Data.TensorDataset(X_train, Y_train)
X_test = torch.from_numpy(x_test).type(torch.FloatTensor)
Y_test = torch.from_numpy(y_test).type(torch.LongTensor)
Label_test = Data.TensorDataset(X_test, Y_test)

X_true = torch.from_numpy(x_true).type(torch.FloatTensor)
Y_true = torch.from_numpy(y_true).type(torch.LongTensor)
Label_true = Data.TensorDataset(X_true, Y_true)

label_ture_loader = Data.DataLoader(Label_true, batch_size=64, shuffle=False)


label_train_loader = Data.DataLoader(Label_train, batch_size=64, shuffle=True)
label_test_loader = Data.DataLoader(Label_test, batch_size=64, shuffle=False)

class OneDNN(nn.Module):
    def __init__(self,band):
        super().__init__()
        self.conv1=nn.Conv1d(in_channels=1,out_channels=8,stride=1,kernel_size=5)
        self.r=nn.ReLU()
        self.flat=nn.Flatten(1)
        #bn batch normalize
        self.pool=nn.MaxPool1d(2,stride=2)
        self.conv2=nn.Conv1d(in_channels=8,out_channels=16,kernel_size=5,stride=1)
        self.conv3=nn.Conv1d(in_channels=16,out_channels=32,kernel_size=5,stride=1)
        if args.dataset=='Indian':
            self.l=nn.Linear(1376,96)
        if args.dataset == 'Pavia':
            self.l = nn.Linear(576, 96)
        if args.dataset == 'Houston':
            self.l = nn.Linear(928, 96)
        self.ll=nn.Linear(96,16)

    def forward(self,input):
        output=self.conv1(input)
        output=self.r(output)
        output=self.pool(output)
        output=self.conv2(output)
        output=self.r(output)
        output=self.pool(output)
        output=self.conv3(output)
        output=self.r(output)
        output=self.flat(output)
        output=self.l(output)
        output=self.ll(output)
        return output

NN=OneDNN(band)
NN=NN.cuda()
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.cuda()
optimizer=torch.optim.Adam(NN.parameters(),lr=0.01)

total_train_step=0


for i in range(args.epoch):

    total_train_step=0
   # print("第{}轮开始".format(i+1))
    for data in label_train_loader:
       # print("{}".format(total_train_step))
        d,target=data
        #神经网络的输入是（batch，channel，height，width）
        #对于1d的卷积，则要求输入是（batch，1，channels）
        d = torch.unsqueeze(d, 1)
        d=d.cuda()
        target=target.cuda()
        outputs=NN(d)
        #outputs的格式应该为batch，类别，对象数（在此任务中为1）
        outputs=torch.unsqueeze(outputs,2)
        target=torch.unsqueeze(target,1)
        temp=torch.ones((target.shape[0],1)).type(torch.LongTensor).cuda()
        target=target.sub(temp)
        loss = loss_fn(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("训练次数{}".format(total_train_step+1))
        total_train_step+=1



    if (i + 1 == args.epoch):
        pree  = torch.zeros(y_true.shape).type(torch.LongTensor)
        k = 0
        with torch.no_grad():
            for data in label_ture_loader:
                d, targets = data
                d = torch.unsqueeze(d, 1)
                d = d.cuda()
                targets = targets.cuda()
                outputs = NN(d)
                outputs = torch.unsqueeze(outputs, 2)
                targets = torch.unsqueeze(targets, 1)
                temp = torch.ones((targets.shape[0], 1)).type(torch.LongTensor).cuda()
                targets = targets.sub(temp)
                # 构造pre矩阵
                outputs = outputs.squeeze()
                a = torch.argmax(outputs, dim=1)
                pree[k:k + a.shape[0]] = a
                k += a.shape[0]
                # 将格式转换为ndarry
            pree = pree.type(torch.LongTensor).numpy()
            t = np.ones(pree.shape)
            pree = np.add(pree, t)

            utils.save_result(pree, width, height, total_pos_true, number_true, "1D-Houston")
        # 跑完一轮之后，要用测试数据集评估一下训练之后的结果
        # with表示不会对源模型的参数做任何的修改
    elif (i+1)%100==0:
        print("Epoch:{}".format(i+1))
        total_test_loss = 0
        pre = torch.zeros(size=y_test.shape)
        k = 0
        with torch.no_grad():
            for data in label_test_loader:

                d, targets = data
                d = torch.unsqueeze(d, 1)
                d = d.cuda()
                targets = targets.cuda()
                outputs = NN(d)
                outputs = torch.unsqueeze(outputs, 2)
                targets = torch.unsqueeze(targets, 1)
                temp = torch.ones((targets.shape[0], 1)).type(torch.LongTensor).cuda()
                targets = targets.sub(temp)
                loss = loss_fn(outputs, targets)
                # item函数把tensor变成数字
                total_test_loss += loss.item()

                #构造pre矩阵
                outputs=outputs.squeeze()
                a=torch.argmax(outputs,dim=1)
                pre[k:k+a.shape[0]]=a
                k+=a.shape[0]
            #将格式转换为ndarry
            pre=pre.type(torch.LongTensor).numpy()
            t=np.ones(pre.shape)
            pre=np.add(pre,t)
            matrix = confusion_matrix(y_test, pre)

            OA, AA_mean, Kappa, AA = utils.cal_results(matrix)
            print()
            print("Overall metrics:")
            print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA_mean, Kappa))
            print("total_loss:{:.4f}".format(total_test_loss))
            utils.cal_class_results(matrix,y_test)

