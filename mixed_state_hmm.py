import numpy as np

class GaussianState():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def sample1(self):
        return float((np.random.normal(self.mean, self.std, 1))[0])



class BernoulliState():

    def __init__(self, prob):
        self.p = prob
    
    def sample1(self):
        return float((np.random.choice([0, 1], 1, p=[1-self.p, self.p]))[0])


class MixedState():

    def __init__(self, mean, std, prob):
        self.mean = mean
        self.std = std
        self.p = prob
        self.normal_state = GaussianState(self.mean, self.std)
        self.bern_state = BernoulliState(self.p)
    
    def sample1(self):
        return [self.normal_state.sample1(), self.bern_state.sample1()]


class MixedHMM():
    
    def __init__(self, mixed_states, trans_mat=None, init_probs=None):
        self.states = [MixedState(*seq) for seq in mixed_states]
        self.num_states = len(mixed_states)

        if init_probs is None:
            self.init_probs = [1.0, 0.0, 0.0]
        
        if trans_mat is None:
            self.trans_mat = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        

        self.curr_state = np.random.choice(list(range(self.num_states)), p=self.init_probs)

    
    def sample_traj(self, n):
        traj_states = [self.curr_state]
        traj_obs = [self.states[self.curr_state].sample1()]
        for t in range(1,n):
            prob = self.trans_mat[self.curr_state]
            self.curr_state = np.random.choice(list(range(self.num_states)), p=self.trans_mat[self.curr_state])
            traj_states.append(self.curr_state)
            traj_obs.append(self.states[self.curr_state].sample1())
        
        return traj_obs, traj_states




if __name__ == "__main__":
    mixed_states = [[0.0, 1.0, 0.5], [10.0, 1.0, 0.5], [20.0, 1.0, 0.5]]
    mhmm = MixedHMM(mixed_states)
    
    print(mhmm.sample_traj(10))






