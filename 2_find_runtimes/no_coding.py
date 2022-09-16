import numpy as np


class NoCoding:
    
    def __init__(self, n, n_jobs, mu, delays) -> None:
        # design parameters
        self.n = n
        self.n_jobs = n_jobs
        self.mu = mu
        
        # parameters
        self.load = self.normalized_load(n)
        self.T = self.delay()
        self.total_rounds = n_jobs + self.T
        
        # delay profile
        assert delays.shape[1] >= n_jobs, \
            'delays should have at least `rounds` elements.' 
        assert delays.shape[0] >= n, \
            'delays.shape[0] should have at least `n` elements'
        self.delays = delays[:n, :self.total_rounds] # (n, rounds)
        
        # state of the master: (worker, minitask, round)
        self.state = np.full((n, self.total_rounds), np.nan) 
        self.durations = np.full((self.total_rounds, ), -1.)
    
    
    @classmethod
    def normalized_load(cls, n):
        return 1 / n
    
    
    @classmethod
    def param_combinations(cls, n, rounds=None, buffer_rounds=None):
        yield tuple()


    @classmethod
    def delay(cls):
        return 0

    
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
        
        # wait for all the workers
        delay = self.delays[:, round_]
        round_duration = delay.max()
            
        # set round_result into state
        self.state[:, round_] = round_result
        self.durations[round_] = round_duration


    def is_decodable(self, r) -> bool:
        """
        To be able to decode: always!
            r (int): round index
        """
        return True
