import numpy as np

class GaussianState():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def sample1(self):
        return float(np.random.normal(self.mean, self.std, 1))



class BernoulliState():

    def __init__(self, prob):
        self.p = prob
    
    def sample1(self):
        float(np.random.choice([0, 1], 1, p=[1-self.prob, self.prob]))


class MixedState():

    def __init__(self, mean, std, prob):
        self.mean = mean
        self.std = std
        self.p = prob
        self.normal_state = GaussianState(self.mean, self.std)
        self.bern_state = BernoulliState(self.p)
    
    def sample1(self):
        return [self.normal_state.sample1(), self.bern_state()]



