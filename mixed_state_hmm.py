import numpy as np
import scipy.stats as stats
import itertools

class GaussianState():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def sample1(self):
        return float((np.random.normal(self.mean, self.std, 1))[0])
    
    def ll(self, x):
        return np.log(stats.norm.pdf(x, loc=self.mean, scale=self.std))


class BernoulliState():

    def __init__(self, prob):
        self.p = prob
    
    def sample1(self):
        return float((np.random.choice([0, 1], 1, p=[1-self.p, self.p]))[0])
    
    def ll(self, x):
        assert (x == 1) or (x==0)
        return np.log((self.p*x) + (1-self.p)*(1-x))


class MixedState():

    def __init__(self, mean, std, prob):
        self.mean = mean
        self.std = std
        self.p = prob
        self.normal_state = GaussianState(self.mean, self.std)
        self.bern_state = BernoulliState(self.p)
    
    def sample1(self):
        return [self.normal_state.sample1(), self.bern_state.sample1()]
    
    def ll(self, x, y):
        return self.normal_state.ll(x) + self.bern_state.ll(y)


class MixedHMM():
    
    def __init__(self, mixed_states, trans_mat=None, init_probs=None):
        self.states = [MixedState(*seq) for seq in mixed_states]
        self.num_states = len(mixed_states)

        if init_probs is None:
            self.init_probs = [1.0, 0.0, 0.0]
        
        if trans_mat is None:
            self.trans_mat = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        
        if trans_mat == "random":
            trans_mat = np.random.rand(self.num_states, self.num_states)
            trans_sum = np.sum(trans_mat, axis=1)
            trans_sum = trans_sum[:, np.newaxis]
            self.trans_mat = trans_mat / trans_sum
        
        if init_probs == "random":
            init_probs = np.random.rand(1, self.num_states)
            self.init_probs = init_probs / sum(init_probs)

        

        self.curr_state = np.random.choice(list(range(self.num_states)), p=self.init_probs)

    
    def sample_full(self, n):
        self.curr_state = np.random.choice(list(range(self.num_states)), p=self.init_probs)
        traj_states = [self.curr_state]
        traj_obs = [self.states[self.curr_state].sample1()]
        for t in range(1,n):
            prob = self.trans_mat[self.curr_state]
            self.curr_state = np.random.choice(list(range(self.num_states)), p=self.trans_mat[self.curr_state])
            traj_states.append(self.curr_state)
            traj_obs.append(self.states[self.curr_state].sample1())
        
        return traj_obs, traj_states
    

    def sample_obs(self, n):
        return self.sample_full(n)[0]
    
    def sample_states(self, n):
        return self.sample_full(n)[1]


    def all_latent_traj(self, n):
        all_traj = [list(range(self.num_states)) for _ in range(n)]
        return list(itertools.product(*all_traj))


    def ll(self, seq):
        log_ll = 0
        for i, (x, y) in enumerate(seq):
            state = self.states[i % self.num_states]
            log_ll += state.ll(x, y)

        return log_ll


        # for element in itertools.product(*somelists)

    
    def avg_ll(self, seqs):
        all_log_ll = 0
        for seq in seqs:
            all_log_ll += self.ll(seq)
        
        return all_log_ll / len(seqs)


    def true_ll(self, seq):
        ll = -np.inf
        seq_len = len(seq)
        for traj in self.all_latent_traj(seq_len):
            traj_ll = 0
            traj_ll += np.log(self.init_probs[traj[0]])
            curr_state = self.states[traj[0]]
            x, y = seq[0]
            traj_ll += curr_state.ll(x,y)
            for (i, (x, y)) in zip(list(range(1, len(traj))), seq[1:]): # zips togethter iterator over trajectory and observations
                traj_ll += np.log(self.trans_mat[traj[i-1], traj[i]])
                curr_state = self.states[traj[i]]
                traj_ll += curr_state.ll(x,y)
            
            ll = np.log(np.exp(ll) + np.exp(traj_ll))

        return ll 
    

    
    def true_avg_ll(self, seqs):
        all_log_ll = 0
        for seq in seqs:
            all_log_ll += self.true_ll(seq)
        
        return all_log_ll / len(seqs)


            





if __name__ == "__main__":
    mixed_states = [[0.0, 1.0, 0.2], [2.0, 1.0, 0.3], [3.0, 1.0, 0.4]]
    mhmm = MixedHMM(mixed_states, trans_mat="random")
    num_time_steps_mix = 2
    test_len_mix = 5
    test_data_mix = np.array([mhmm.sample_obs(num_time_steps_mix) for _ in range(test_len_mix)])
    avg_ll_mix = mhmm.avg_ll(test_data_mix)
    print(mhmm.avg_ll(test_data_mix))
    print(mhmm.true_avg_ll(test_data_mix))






