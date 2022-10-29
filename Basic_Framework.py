import numpy as np
import scipy.special  # sigmoid function


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
        # 输入数据为行向量，注意特定训练集时需要转换
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
            hidden_error[i] = np.dot(np.array(self.whh[i]).T, hidden_error[i + 1])
        input_error = np.dot(np.array(self.wih).T, hidden_error[0])

        self.who += self.learning_rate * np.dot((final_output * (1 - final_output) * final_error),
                                                np.array(hidden_output[-1], ndmin=2).T)
        self.wih += self.learning_rate * np.dot((hidden_output[0] * (1 - hidden_output[0]) * hidden_error[0]),
                                                np.array(input_data, ndmin=2).T)
        for i in range(0, self.hidden_layers - 2):
            self.whh[i] += self.learning_rate * np.dot(
                (hidden_output[i + 1] * (1 - hidden_output[i + 1]) * hidden_error[i + 1]),
                np.array(hidden_output[i], ndmin=2).T)
