import argparse
import utils
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA

parser=argparse.ArgumentParser("2DNN")
parser.add_argument('dataset',choices=['Indian','Houston','Pavia'],default='Indian')
parser.add_argument('epoch', type=int, default=500, help='epoch number')
parser.add_argument('patch',type=int,help='size of input',default=7)
parser.add_argument('usePCA',type=bool,default=False)
args=parser.parse_args()

data,data_TE,data_TR,data_label=utils.loaddata(args.dataset)

input_normalize=utils.normalize(data)

total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = utils.chooose_train_and_test_point(
    data_TR, data_TE, data_label, 16)
height,width,band=input_normalize.shape
#对于2D的CNN，我们需要对于获取其周边的像素
#在1DCNN当中，我们传入的数据的形状是【x,200】,其中x为全部打散的波段
#我们现在希望传入输入的大小是【x，width，height，band】
#再对于每一张小图匹配一个标签

if args.usePCA==True:
    #n_components表示的是期待将为之后的维度，band从200将为30
    pca=PCA(n_components=30)
    input_normalize=input_normalize.reshape(-1,band)
    input_normalize=pca.fit_transform(input_normalize)
    input_normalize=input_normalize.reshape(height,width,-1)
    band = 30
#在测试当中我们向网络输入7*7的小patch，这意味着我们需要对于原图像的边界进行3的像素的padding
#我们在这里不使用0来填充，我们使用了镜像的填充方法

#调用utils里面的mirror方法来实现这个功能
#对于Indian，原输入是145*145*200，左边增加三个像素右边增加三个像素，最终的结果就是151*151*200
#input_mirror=utils.padWithZeros(height, width, band, input_normalize,(int)(args.patch/2))
input_mirror=utils.mirror(height, width, band, input_normalize, patch=args.patch)

#获取每一个周围的像素，大小为【x，7，7，200】
print("开始读取数据")
x_train,x_test,x_true=utils.train_and_test_data(input_mirror,band,total_pos_train,total_pos_test,total_pos_true,patch=args.patch)

y_train, y_test, y_true = utils.toStandardformY(data_label, band, total_pos_train, total_pos_test, total_pos_true,
                                          number_train, number_test, number_true)

# 将numpy的ndarray类型转换为torch的tensor类型，再用tensordataset读取
#torch.set_default_dtype(torch.float16)
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

        #三层三维卷积
        self.conv1=nn.Sequential(
            nn.Conv3d(in_channels=1,out_channels=8,kernel_size=(7,3,3),stride=1,padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv2=nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3=nn.Sequential(
            nn.Conv3d(in_channels=16,out_channels=32,kernel_size=(3,3,3),stride=1,padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        #一层二维卷积
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=576,out_channels=64,kernel_size=(3,3),stride=1,padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.flat=nn.Flatten(1)


        if args.patch == 15:
            self.l = nn.Linear(3136,256)
        if args.patch == 17:
            self.l = nn.Linear(5184, 256)
        if args.patch == 19:
            self.l = nn.Linear(7744, 256)
        if args.patch == 21:
            self.l = nn.Linear(10816, 256)
        if args.patch == 23:
            self.l = nn.Linear(14400, 256)
        if args.patch == 25:
            self.l=nn.Linear(18496,256)

        self.ll = nn.Linear(256, 128)
        self.lll = nn.Linear(128, 16)

        self.r = nn.ReLU()
        self.drop = nn.Dropout(0.4)

    def forward(self,input):
        output=self.conv1(input)
        output=self.conv2(output)
        output=self.conv3(output)

        #reshape

        output = output.reshape(output.shape[0], -1,output.shape[3], output.shape[3])
        output=self.conv4(output)

        output=self.flat(output)

        output=self.l(output)
        output=self.r(output)
        output=self.drop(output)

        output=self.ll(output)
        output = self.r(output)
        output = self.drop(output)

        output=self.lll(output)
        return output

NN=OneDNN(band)
NN=NN.cuda()
NN=NN
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.cuda()
optimizer=torch.optim.Adam(NN.parameters(), lr=0.001)

total_train_step=0

print("Start train")
for i in range(args.epoch):

    total_train_step=0
    NN.train()
   # print("第{}轮开始".format(i+1))
    for data in label_train_loader:
        #print("{}".format(total_train_step))
        d,target=data
        #神经网络的输入是（batch，channel，band,height，width）
        #permute 按序号交换维度
        d=d.permute(0,3,1,2)
        d=d.unsqueeze(1)
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
        pree = torch.zeros(y_true.shape).type(torch.LongTensor)
        k = 0
        NN.eval()
        with torch.no_grad():
            for data in label_ture_loader:
                d, targets = data
                d = d.permute(0, 3, 1, 2)
                d = d.unsqueeze(1)
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

            utils.save_result(pree, height, width, total_pos_true, number_true, "HybridSN-Houston")
        # 跑完一轮之后，要用测试数据集评估一下训练之后的结果
        # with表示不会对源模型的参数做任何的修改
    if (i+1)%10==0:
        print("Epoch:{}".format(i+1))
        total_test_loss = 0
        pre = torch.zeros(size=y_test.shape)
        k = 0
        NN.eval()
        with torch.no_grad():
            for data in label_test_loader:

                d, targets = data
                d = d.permute(0, 3, 1, 2)
                d = d.unsqueeze(1)
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
