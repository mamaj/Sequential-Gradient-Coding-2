import numpy as np


class GradientCoding:
    
    def __init__(self, n, s, rounds, mu, delays) -> None:
        # design parameters
        self.n = n
        self.s = s
        self.rounds = rounds
        self.mu = mu
        
        # parameters
        self.load = self.normalized_load(n, s)
        self.total_rounds = rounds
        
        # delay profile
        assert delays.shape[1] >= rounds, \
            'delays should have at least `rounds` elements.' 
        assert delays.shape[0] >= n, \
            'delays.shape[0] should have at least `n` elements'
        self.delays = delays[:n, :self.total_rounds] # (n, rounds)
        
        # state of the master: (worker, minitask, round)
        self.state = np.full((n, self.total_rounds), np.nan) 
        self.durations = np.full((self.total_rounds, ), -1.)
    
    
    @classmethod
    def normalized_load(cls, n, s):
        return (s + 1) / n
    
    
    @classmethod
    def param_combinations(cls, n, rounds=None, buffer_rounds=None):
        for s in range(1, n+1):
            yield (s, )

    
    
    def run(self) -> None:
        for round_ in range(self.total_rounds):
            # perform round
            self.perform_round(round_)
            
            if not self.is_decodable(round_):
                raise RuntimeError(f'round {round_} is not decodable.')
                 
    
    def perform_round(self, round_) -> None:
        """ This will fill state(:, round_)
        """
        round_result = np.full((self.n, ), round_)
        
        # apply stragglers
        delay = self.delays[:, round_]
        wait_time = delay.min() * (1 + self.mu) 
        is_straggler = delay > wait_time
        
        if self.follows_straggler_model(is_straggler):
            # do not wait for all: apply straggler pattern
            round_result[is_straggler] = -1
            round_duration = wait_time
        else:
            # wait for all: do not apply stragglers
            round_duration = delay.max()
            
        # set round_result into state
        self.state[:, round_] = round_result
        self.durations[round_] = round_duration


    def is_decodable(self, r) -> bool:
        """
        To be able to decode: there should be less than s straggelers.
            r (int): round index
        """
        num_stragglers = (self.state[:, r] == -1).sum()
        return num_stragglers < self.s
    
    
    def follows_straggler_model(self, is_straggler) -> bool:
        """ Checks if at any given round, if s-stragglers-per-round condition
            is met.
            
            r (int): current round idx.
            is_straggler (ndarray): boolean array of length n.
        """
        
        return is_straggler.sum() < self.s
