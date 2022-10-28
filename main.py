import numpy as np
import scipy.special
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio.v2 as imageio
mpl.use('TkAgg')

print("Initializing, please wait a moment...")


class NeuralNetwork:
    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes  # 设置输入节点数量
        self.hnodes = hiddennodes  # 设置隐藏层节点数量
        self.onodes = outputnodes  # 设置输出节点数量
        self.lr = learningrate  # 设置学习等级

        # 随机设置权重，随机返回一个numpy数组
        self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)  # 设置入-隐权重
        self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)  # 设置隐-出权重

        # 创建激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):  # 训练神经网络
        # 计算误差
        inputs = np.array(inputs_list, ndmin=2).T  # 倒置输入矩阵
        targets = np.array(targets_list, ndmin=2).T  # 倒置目标矩阵
        hidden_inputs = np.dot(self.wih, inputs)  # 用入-隐权重乘以输入矩阵，得到隐藏层矩阵
        hidden_outputs = self.activation_function(hidden_inputs)  # 对隐藏层使用激活函数，得到输出层矩阵
        final_inputs = np.dot(self.who, hidden_outputs)  # 用隐-出权重乘以输出层矩阵，得到最终输入矩阵
        final_outputs = self.activation_function(final_inputs)  # 对最终输入矩阵使用激活函数，得到最终输出矩阵

        output_errors = targets - final_outputs  # 计算输出层误差
        hidden_errors = np.dot(self.who.T, output_errors)  # 计算隐藏层误差

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        np.transpose(hidden_outputs))  # 更新隐-出权重

        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        np.transpose(inputs))  # 更新入-隐权重

    def query(self, inputs_list):  # 识别图像
        # 进行计算，同train方法中的内容
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# 设置神经网络规模
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.3

# 读取训练数据
training_data_file = open("mnist_train.csv")  # 打开文件
training_data_list = training_data_file.readlines()  # 逐行读取，在训练数据集中，一行数据就是一张图片
training_data_file.close()  # 关闭文件

# 创建实例，并对神经网络使用训练集数据来训练
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
for record in training_data_list:
    all_values = record.split(',')  # 以“，”为分隔，读取训练数据集中的数据。训练数据集以第一个元素作为索引
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # 将颜色从较大的0-255范围缩小到0.01-1.0的范围
    targets = np.zeros(output_nodes) + 0.01  # 初始化输出节点，将所有值用0填充后加上0.01
    targets[int(all_values[0])] = 0.99  # 将列表目标的正确元素设置为0.99
    n.train(inputs, targets)

# 读取要识别的图像并处理
Useriuput = 'y'
while Useriuput == 'y':
    img_array = imageio.imread("/home/kana/下载/daw.png", pilmode='F')
    img_data = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01

    fiout = n.query(img_data)
    label = np.argmax(fiout)
    print(f"Neural Network think the number is {label}.")
    print(fiout)
    print('\n')
    plt.imshow(img_array, cmap='Greys', interpolation='None')
    plt.show()

    Useriuput = input("Enter y to continue identification, otherwise exit: ")
