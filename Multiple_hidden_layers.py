import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import Basic_Framework
mpl.use('TkAgg')


inputNodes = 784
hiddenNodes = 100
outputNodes = 10
hiddenLayers = 3
learningRate = 0.003
epoch = 1

Network1 = Basic_Framework.BPNN(inputNodes, hiddenNodes, outputNodes, hiddenLayers, learningRate)


training_data_file = open("../DataSets/mnist_train.csv")
test_data_file = open('../DataSets/mnist_test.csv')
training_data_list = training_data_file.readlines()  # In the training dataset, a row of data is an image
test_data_list = test_data_file.readlines()
training_data_file.close()
test_data_file.close()


input_list = []
target_list = []
for record in training_data_list:
    all_values = record.split(',')  # 以“，”为分隔，读取训练数据集中的数据。训练数据集以第一个元素作为索引
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # 将颜色从较大的0-255范围缩小到0.01-1.0的范围
    targets = np.zeros(outputNodes) + 0.01  # 初始化输出节点，将所有值用0填充后加上0.01
    targets[int(all_values[0])] = 0.99  # 将列表目标的正确元素设置为0.99
    input_list.append(inputs)
    target_list.append(targets)

for i in range(0, epoch):
    for j in range(0, len(input_list)):
        Network1.train(input_list[j], target_list[j])
    Right = int(0)
    for line in test_data_list:
        all_values = line.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        fiout = Network1.query(inputs)
        label = np.argmax(fiout)
        if int(label) == int(all_values[0]):
            Right += 1
    print(f"{i+1} : {Right}%")

userinput = 'x'
while userinput == 'y':
    img_array = imageio.imread("/home/kana/下载/5.png", pilmode='F')
    img_data = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01

    fiout = Network1.query(img_data)

    label = np.argmax(fiout)
    print(label)
    print(fiout)
    plt.imshow(img_array, cmap='Greys', interpolation='None')
    plt.show()
    print()
    userinput = input("INPUT: ")

