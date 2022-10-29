import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import scipy.special
import softmax
mpl.use('TkAgg')


class BPNN:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, hidden_layers, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate

        self.sigmoid = lambda x: scipy.special.expit(x)

        self.whh = []
        for i in range(1, hidden_layers):
            self.whh.append(np.random.rand(hidden_nodes, hidden_nodes) - 0.5)
        self.wih = (np.random.rand(hidden_nodes, input_nodes) - 0.5)
        self.who = (np.random.rand(output_nodes, hidden_nodes) - 0.5)

    def query(self, input_data):
        # 输入数据为维维行向量，注意特定训练集时需要转换
        input_data = np.array(input_data, ndmin=2).T
        hidden_input = []
        hidden_output = []
        hidden_input.append(np.dot(self.wih, input_data))
        for i in range(0, self.hidden_layers - 1):
            hidden_output.append(self.sigmoid(hidden_input[i]))
            hidden_input.append(np.dot(self.whh[i], hidden_output[i]))
        hidden_output.append(self.sigmoid(hidden_input[-1]))
        final_input = np.dot(self.who, hidden_output[-1])
        final_output = self.sigmoid(final_input)
        return final_output

    def train(self, input_data, target_data):
        # 注意：输入数据和目标数据均为行向量，必要时需要进行矩阵转置
        input_data = np.array(input_data, ndmin=2).T
        target_data = np.array(target_data, ndmin=2).T
        hidden_input = []
        hidden_output = []
        hidden_input.append(np.dot(self.wih, input_data))
        for i in range(0, self.hidden_layers - 1):
            hidden_output.append(self.sigmoid(hidden_input[i]))
            hidden_input.append(np.dot(self.whh[i], hidden_output[i]))
        hidden_output.append(self.sigmoid(hidden_input[-1]))
        final_input = np.dot(self.who, hidden_output[-1])
        final_output = self.sigmoid(final_input)

        final_error = target_data - final_output
        hidden_error = []
        for i in range(0, self.hidden_layers):
            hidden_error.append(np.array([1, 2]))
        hidden_error[self.hidden_layers - 1] = np.dot(np.array(self.who).T, final_error)
        for i in range(self.hidden_layers - 2, -1, -1):
            hidden_error[i] = np.dot(np.array(self.whh[i]).T, hidden_error[i+1])
        input_error = np.dot(np.array(self.wih).T, hidden_error[0])

        self.who += self.learning_rate * np.dot((final_output * (1 - final_output) * final_error),
                                                np.array(hidden_output[-1], ndmin=2).T)
        self.wih += self.learning_rate * np.dot((hidden_output[0] * (1 - hidden_output[0]) * hidden_error[0]),
                                                np.array(input_data, ndmin=2). T)
        for i in range(0, self.hidden_layers - 2):
            self.whh[i] += self.learning_rate * np.dot((hidden_output[i+1] * (1-hidden_output[i+1])*hidden_error[i+1]),
                                                        np.array(hidden_output[i], ndmin=2).T)


inputnodes = 784
hiddennodes = 100
outputnodes = 10
hiddenlayers = 3
learningrate = 0.003
epoch = 1

training_data_file = open("/home/kana/Desktop/Python_Test/DataBase/mnist_train.csv")  # 打开文件
training_data_list = training_data_file.readlines()  # 逐行读取，在训练数据集中，一行数据就是一张图片
training_data_file.close()  # 关闭文件

test_data_file = open('/home/kana/Desktop/Python_Test/DataBase/mnist_test.csv')
test_data_list = test_data_file.readlines()
test_data_file.close()

myNN = BPNN(inputnodes, hiddennodes, outputnodes, hiddenlayers, learningrate)

input_list = []
target_list = []
for record in training_data_list:
    all_values = record.split(',')  # 以“，”为分隔，读取训练数据集中的数据。训练数据集以第一个元素作为索引
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # 将颜色从较大的0-255范围缩小到0.01-1.0的范围
    targets = np.zeros(outputnodes) + 0.01  # 初始化输出节点，将所有值用0填充后加上0.01
    targets[int(all_values[0])] = 0.99  # 将列表目标的正确元素设置为0.99
    input_list.append(inputs)
    target_list.append(targets)

for i in range(0, epoch):
    for j in range(0, len(input_list)):
        myNN.train(input_list[j], target_list[j])
    Right = int(0)
    for line in test_data_list:
        all_values = line.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        fiout = myNN.query(inputs)
        rfiout = np.zeros([10, 1])
        for i in range(0, 10):
            pass
        label = np.argmax(fiout)
        if int(label) == int(all_values[0]):
            Right += 1
    print(f"{i+1} : {Right}%")

userinput = 'y'
while userinput == 'y':
    img_array = imageio.imread("/home/kana/下载/5.png", pilmode='F')
    img_data = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01

    fiout = myNN.query(img_data)
    label = np.argmax(fiout)
    print(label)
    print(fiout)
    plt.imshow(img_array, cmap='Greys', interpolation='None')
    plt.show()
    print()
    userinput = input("INPUT: ")

