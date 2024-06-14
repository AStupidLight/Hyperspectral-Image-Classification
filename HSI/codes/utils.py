import numpy as np
from scipy.io import loadmat
import torch
import torch.utils.data as Data
from scipy.io import savemat

# 移动文件的库
import shutil


# 计算OA，AA和Kappa
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def cal_class_results(matrix, y):
    x = np.sum(matrix, axis=1)
    for i in range(max(y)):
        accuracy = matrix[i][i] / x[i]
        print("Class:{}".format(i + 1))
        print("Accuracy:{:.2f}".format(accuracy * 100))


def save_result(pre, width, height, total_pos_test, number_test, name):
    classification_result = np.zeros(shape=(width, height), dtype=int)
    for i in range(sum(number_test)):
        classification_result[total_pos_test[i][0], total_pos_test[i][1]] += pre[i]
    savemat(name + '.mat', mdict={"result": classification_result})
    shutil.move(name + '.mat', './results')


def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    # TR, TE, label, num_classes
    # train_data和true_data都是只保存标签信息

    # number_train 每个类别的总数，类别代表数量
    number_train = []
    # pos_train用一个字典表示每一个类别对应的坐标，key是类别序号，键值是一个{x，2}的矩阵，每一行代表一个坐标
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    # -------------------------for train data------------------------------------
    # 统计每种种类的数量
    for i in range(num_classes):
        each_class = []
        # 标签等于每种种类，不吃背景label=0，只吃实际转换的大小
        each_class = np.argwhere(train_data == (i + 1))
        # 在number_train后面加上这个类别的数量，shape【0】代表着数量
        number_train.append(each_class.shape[0])
        # 用字典保存所有的坐标
        pos_train[i] = each_class
    # 要构造一个最终的全部的坐标点，初始化为标签为0的一些点
    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        # 把别的点都接在后面
        total_pos_train = np.r_[total_pos_train, pos_train[i]]  # (695,2)
    # 把类型改变为int
    total_pos_train = total_pos_train.astype(int)
    # --------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data == (i + 1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]  # (9671,2)
    total_pos_test = total_pos_test.astype(int)
    # --------------------------for true data------------------------------------
    for i in range(num_classes + 1):
        each_class = []
        each_class = np.argwhere(true_data == i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes + 1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true


def toStandardformX(input_normalize, band, total_pos_train, total_pos_test, total_pos_true, number_train, number_test,
                    number_true):
    X_train = np.zeros(shape=(sum(number_train), band), dtype=float)
    X_test = np.zeros(shape=(sum(number_test), band), dtype=float)
    X_true = np.zeros(shape=(sum(number_true), band), dtype=float)
    for i in range(sum(number_train)):
        X_train[i, :] = input_normalize[total_pos_train[i][0]][total_pos_train[i][1]]
    for i in range(sum(number_test)):
        X_test[i, :] = input_normalize[total_pos_test[i][0]][total_pos_test[i][1]]
    for i in range(sum(number_true)):
        X_true[i,:]=input_normalize[total_pos_true[i][0]][total_pos_true[i][1]]
    return X_train, X_test, X_true


def toStandardformY(data_label, band, total_pos_train, total_pos_test, total_pos_true, number_train, number_test,
                    number_true):
    Y_test = np.zeros(shape=(sum(number_test)), dtype=int)
    Y_train = np.zeros(shape=(sum(number_train)), dtype=int)
    Y_true = np.zeros(shape=(sum(number_true)), dtype=int)
    for i in range(sum(number_train)):
        Y_train[i] = data_label[total_pos_train[i][0]][total_pos_train[i][1]]
    for i in range(sum(number_test)):
        Y_test[i] = data_label[total_pos_test[i][0]][total_pos_test[i][1]]
    for i in range(sum(number_train)):
        Y_true[i] = data_label[total_pos_test[i][0]][total_pos_test[i][1]]
    return Y_train, Y_test, Y_true


def loadColormap(dataset):
    global temp
    if dataset == 'Indian':
        temp = loadmat('./colormaps/colormapIndian.mat')
    elif dataset == 'Houston':
        temp = loadmat('colormaps/colormapHouston.mat')
    elif dataset == 'Pavia':
        temp = loadmat('./colormaps/colormapPavia.mat')
    return temp


def loaddata(dataset):
    global da
    if dataset == 'Indian':
        # loadmat以字典的形式保存数据
        da = loadmat('./datasets/IndianPine.mat')
    elif dataset == 'Houston':
        da = loadmat('./datasets/Houston.mat')
    elif dataset == 'Pavia':
        da = loadmat("./datasets/Pavia.mat")
    data = da['input']  # 取出实际使用的数据
    data_TE = da['TE']
    data_TR = da['TR']
    data_label = data_TE + data_TR
    return data, data_TE, data_TR, data_label


def normalize(data):
    # 创建一个同样大小的矩阵用于保存数据
    input_normalize = np.zeros(data.shape)
    # data。shape【2】指的是对于每一个波段来做标准化，对于每一个波段，找最大最小值，再归一化
    for i in range(data.shape[2]):
        input_max = np.max(data[:, :, i])
        input_min = np.min(data[:, :, i])
        input_normalize[:, :, i] = (data[:, :, i] - input_min) / (input_max - input_min)

    # number_train==[50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 15, 15, 15]
    # number_test==[1384, 784, 184, 447, 697, 439, 918, 2418, 564, 162, 1244, 330, 45, 39, 11, 5]
    # number_true==[10659, 1434, 834, 234, 497, 747, 489, 968, 2468, 614, 212, 1294, 380, 95, 54, 26, 20]
    # number_true多一个维数是背景像素，10659个

    # sum(number_train)==695，恰好等于total_pos_train里元素的数量

    return input_normalize


def mirror(height, width, band, input_normalize, patch=5):
    # 截断除法，除完之后向下取整，并且返回的是一个int值
    padding = patch // 2
    # 对图片进行一个扩大化，在边界上扩增一个padding的长度，左边和右边各一个padding，最终总长增加了两个padding
    # padding上的值相当于将中间的部分折到边上
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    # 中心区域
    # 中心区域的值直接等于原来的部分
    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize
    # 左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    # 右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    # 上边镜像
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]
    # 下边镜像
    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi


def padWithZeros(height, width, band, input_normalize, patch=5):
    newX = np.zeros((height + 2 * patch, width + 2 * patch, band))
    x_offset = patch
    y_offset = patch
    newX[x_offset:height + x_offset, y_offset:width + y_offset, :] = input_normalize
    return newX


def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    # 在这个点的右下角取一个立方体
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image


# -------------------------------------------------------------------------------
# 汇总训练数据和测试数据mirror_image, band, total_pos_train, total_pos_test total_pos_true, patch=args.patches, band_patch=args.band_patches
def train_and_test_data(mirror_image, band, train_point, test_point,true_point, patch=5):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype='float16')
    # 获取右下角的patch*patch个像素以及200波段
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for j in range(true_point.shape[0]):
        x_true[j, :, :, :] = gain_neighborhood_pixel(mirror_image, true_point, j, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))

    return x_train, x_test,x_true

# def make_results(NN,label_ture_loader,pree):
