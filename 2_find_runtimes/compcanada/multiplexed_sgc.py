import numpy as np


class MultiplexedSGC:
    
    def __init__(self, n, B, W, lambd, rounds, mu, delays) -> None:
        # design parameters
        self.n = n
        self.B = B
        self.W = W
        self.lambd = lambd
        self.rounds = rounds
        self.mu = mu
        
        # parameters
        self.D1 = (W - 1)
        self.D2 = B
        self.minitasks = W - 1 + B
        self.T = W - 2 + B
        self.load = self.normalized_load(n, B, W, lambd)
        self.total_rounds = rounds + self.T
        
        # delay profile
        assert delays.shape[1] >= self.total_rounds, \
            'delays.shape[1] should have at least `rounds + W-2+B` elements.' 
        assert delays.shape[0] >= n, \
            'delays.shape[0] should have at least `n` elements'
        self.delays = delays[:n, :self.total_rounds] # (n, rounds)
        
        # state of the master: (worker, minitask, round)
        self.state = np.full((n, self.minitasks, self.total_rounds), np.nan) 
        self.durations = np.full((self.total_rounds, ), -1.)
                
        # constants
        self.D1_TOKEN = 0
        self.D2_TOKENS = np.arange(B) + 1 # B tokens, one for each D2 group
    
    
    @classmethod
    def normalized_load(cls, n, B, W, lambd):
        if lambd == n:
            return (W-1+B) / (n * (W-1))
        else:
            return ((lambd+1) * (W-1+B)) / (n * (B + (W-1) * (lambd+1)))
        

    @classmethod
    def param_combinations(cls, n, rounds, max_delay):
        for lambd in range(1, n+1):
            for W in range(2, rounds):
                for B in range(1, W):
                    if max_delay >= W - 1 + B:
                        yield B, W, lambd

    
    def run(self) -> None:
        for round_ in range(self.total_rounds):
            # perform round
            self.perform_round(round_)
            
            # decode
            job = self._get_job(round_)
            if job >= 0 and not self.is_decodable(job):
                raise RuntimeError(f'Job {job} in round {round_} is not decodable.')
                 
    
    def perform_round(self, round_) -> None:
        """ This will fill state(:, :, round_)
        """
        round_result = np.full((self.n, self.minitasks), np.nan)
        
        for m in range(self.minitasks):
            job = self._get_job(round_, m)
            if job < 0:
                break
            
            # fill first D1 minitasks of workers with D1_TOKEN
            if m < self.D1:
                round_result[:, m] = self.D1_TOKEN
            
            # for next minitasks, if D1 of D1_TOKEN is present on diagonal, put 
            # D2_TOKEN of the group, otherwise put D1_TOKEN
            else:
                group = m - self.D1
                num_d1 = (self.task_results(job) == self.D1_TOKEN).sum(axis=1)
                round_result[:, m] = \
                    np.where(num_d1 >= self.D1, self.D2_TOKENS[group], self.D1_TOKEN)
        
        # apply stragglers
        delay = self.delays[:, round_]
        wait_time = delay.min() * (1 + self.mu) 
        is_straggler = delay > wait_time
        
        if self.follows_straggler_model(round_, is_straggler):
            # do not wait for all: apply straggler pattern
            round_result[is_straggler, :] = -1
            round_duration = wait_time
        else:
            # wait for all: do not apply stragglers
            round_duration = delay.max()
            
        # set round_result into state
        self.state[:, :, round_] = round_result
        self.durations[round_] = round_duration


    def _get_job(self, round_, minitask=None) -> int:
        """ returns the job corresponding to a minitask in a round. 
        if minitas is None: 
            returns the job index that is decodable in round round_
        else:
            returns the job that the minitask belongs to
        """
        
        minitask = self.minitasks-1 if minitask is None else minitask
        return round_ - minitask
        

    def is_decodable(self, job) -> bool:
        """
        To be able to decode:
            1. Each worker should have all of its D1 chunks.
            2. In total, at least `n - lambda` coded results from each of the
               B groups in D2.
        """
        
        task_results = self.task_results(job) # (n, minitasks) the diagonals of every worker
        
        # 1. Each worker should have D1 of D1_TOKEN
        num_d1 = (task_results == self.D1_TOKEN).sum(axis=1)
        if np.any(num_d1 < self.D1):
            return False
        
        # 2. There should be at least `lambd` of each D2_TONKENS in task_results
        num_d2 = (task_results.flatten()[:, None] == self.D2_TOKENS).sum(axis=0)
        if np.any(num_d2 < self.n - self.lambd):
            return False
        
        return True
        
    
    def task_results(self, job) -> np.ndarray:
        """ returns the diagonals of every worker for job.
                shape: (n, minitasks) 
                minitasks = W-1 [=D1 slots] + B [=D2 slots]
        """
        
        # axis1 = minitask ax, axis2 = round ax
        return self.state.diagonal(axis1=1, axis2=2, offset=job)
    
    
    def follows_straggler_model(self, r, is_straggler) -> bool:
        """ Checks if at any given round, the spatial and temporal conditions 
            of (B, W, lambd)-bursty straggler model are met.
            
            1- spatial correlation: within the past W rounds, at most `lambd`
            unique stragglers.
            2- temporal correlation: if worker i is a straggler at the current
            round, it cannot be a straggler in [-W, -B] rounds relative to 
            the current round.
            
            r (int): current round idx.
            is_straggler (ndarray): boolean array of length n.
        """
        
        # 1. spatial cond: at most `lambd` unique stragglers over the 
        # past W rounds.
        state_window = self.state[:, 0, np.maximum(0, r+1-self.W) : r]
        been_straggler = (state_window == -1).any(axis=1)
        num_stragglers = (been_straggler | is_straggler).sum()
        
        if num_stragglers > self.lambd:
            return False
        
        # 2. temporal cond: if worif worker i is a straggeler at the 
        # current round, it cannot be a straggeler in [-W, -B]:
        
        state_window = self.state[:, 0, np.maximum(0, r+1-self.W) : np.maximum(0, r+1-self.B)]
        been_straggler = (state_window == -1).any(axis=1)
        
        if (been_straggler & is_straggler).any():
            return False
        
        return True
