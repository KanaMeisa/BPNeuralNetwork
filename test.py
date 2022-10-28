import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.special
import imageio.v2 as imageio
mpl.use('TkAgg')


class BPNN:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inputN = inputnodes
        self.hiddenN = hiddennodes
        self.outputN = outputnodes
        self.learningR = learningrate

        self.sigmoid = lambda x: scipy.special.expit(x)

        self.wih = (np.random.rand(hiddennodes, inputnodes) - 0.5)
        self.who = (np.random.rand(outputnodes, hiddennodes) - 0.5)

    def train(self, inputArray, targetArray):
        inputArray = np.array(inputArray, ndmin=2).T
        hiddenInput = np.dot(inputArray, self.wih)
        hiddenOutput = self.sigmoid(hiddenInput)
        finalInput = np.dot(hiddenOutput, self.who)
        finalOutput = self.sigmoid(finalInput)

        finalError = targetArray - finalOutput
        hiddenError = np.dot(self.who.T, finalError)

        self.who += self.learningR * np.dot(finalError * finalOutput * (1 - finalOutput), np.transpose(hiddenOutput))
        self.wih += self.learningR * np.dot(hiddenError * hiddenOutput * (1 - hiddenOutput), np.transpose(inputArray))

    def query(self, inputArray):
        inputArray = np.array(inputArray, ndmin=2).T
        hiddenInput = np.dot(inputArray, self.wih)
        hiddenOutput = self.sigmoid(hiddenInput)
        finalInput = np.dot(hiddenOutput, self.who)
        finalOutput = self.sigmoid(finalInput)
        return finalOutput


trainDataFile = open("mnist_train.csv")
trainData = trainDataFile.readlines()
trainDataFile.close()

n = BPNN(784, 100, 10, 0.3)
for line in trainData:
    n.train(line)

