import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import Basic_Framework
import softmax

mpl.use('TkAgg')


inputNodes = 784
hiddenNodes = 100
outputNodes = 10
hiddenLayers = 2
learningRate = 0.001
epoch = 50

Network1 = Basic_Framework.BPNeuralNetwork(
    inputNodes, hiddenNodes, outputNodes, hiddenLayers, learningRate)


training_data_file = open("../DataSets/mnist_train.csv")
test_data_file = open('../DataSets/mnist_test.csv')
training_data_list = training_data_file.readlines()  # In the training dataset, a row of data is an image
test_data_list = test_data_file.readlines()
training_data_file.close()
test_data_file.close()


input_list = []
target_list = []
for line in training_data_list:
    all_values = line.split(',')            # 以“,”为分隔，读取训练数据集中的数据。训练数据集以第一个元素作为索引
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # 将颜色从较大的0-255范围缩小到0.01-1.0的范围
    targets = np.zeros(outputNodes) + 0.01  # 初始化输出节点，将所有值用0填充后加上0.01
    targets[int(all_values[0])] = 0.99      # 将列表目标的正确元素设置为0.99

    input_list.append(inputs)
    target_list.append(targets)

userinput1 = 'y'
while userinput1 == 'y':
    for i in range(0, epoch):
        for j in range(0, len(input_list)):
            Network1.train(input_list[j], target_list[j])
        Right = int(0)
        for line in test_data_list:
            all_values = line.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            finalOutput = Network1.query(inputs)
            softmaxOutput = np.zeros([outputNodes, 1])
            for k in range(0, 10):
                softmaxOutput[k, 0] = softmax.softmax(k, finalOutput)
            # label = np.argmax(finalOutput)
            label = np.argmax(softmaxOutput)

            if int(label) == int(all_values[0]):
                Right += 1
        print(f"Epoch {i+1} : {Right}%")
        userinput1 = input()


print()
userinput = 'y'
while userinput == 'y':
    img_array = imageio.imread("TestPic1.png", pilmode='F')
    img_data = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01

    finalOutput = Network1.query(img_data)
    softmaxOutput = np.zeros([outputNodes, 1])
    for k in range(0, 10):
        softmaxOutput[k, 0] = softmax.softmax(k, finalOutput)
    # label = np.argmax(finalOutput)
    label = np.argmax(finalOutput)

    print(label)
    print(finalOutput)

    plt.imshow(img_array, cmap='Greys', interpolation='None')
    plt.show()

    print()
    userinput = input("INPUT: ")
