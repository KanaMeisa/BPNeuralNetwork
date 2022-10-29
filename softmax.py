import math
import numpy


def softmax(target, numbers):
    sums = float(0)
    n = numbers.shape[0]
    for i in range(0, n):
        sums += math.exp(numbers[i, 0])
    answer = math.exp(numbers[target-1, 0]) / sums
    return answer
