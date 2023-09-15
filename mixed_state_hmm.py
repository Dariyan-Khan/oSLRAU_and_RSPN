import numpy as np

class State():

    def __init__(self):
        self.dist=None
    
    def emission(self, length):
        return [self.dist() for _ in range(length)]

class GaussianState(State):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.dist = float(np.random.normal(self.mean, self.std, 1))


class BernoulliState(State):

    def __init__(self, prob):
        self.p = prob
        self.dist = float(np.random.choice([0, 1], 1, p=[1-prob, prob]))



