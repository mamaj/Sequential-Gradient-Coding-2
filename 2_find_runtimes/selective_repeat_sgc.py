import numpy as np
import math

class SelectiveRepeatSGC:
    
    def __init__(self, n, B, W, lambd, n_jobs, mu, delays) -> None:
        # design parameters
        self.n = n # num workers
        self.B = B
        self.W = W
        self.lambd = lambd
        self.n_jobs = n_jobs
        self.mu = mu
        assert (W-1) % B == 0, 'B should devide W-1.'
        
        # parameters
        self.s = math.ceil((B * lambd) / (W - 1 + B))
        self.load = self.normalized_load(n, self.s)
        self.T = self.delay(B, W, lambd)
        self.total_rounds = n_jobs + self.T
        
        # delay profile
        assert delays.shape[1] >= self.total_rounds,\
            'delays.shape[1] should have at least `rounds + B` elements.'
        assert delays.shape[0] >= n, \
            'delays.shape[0] should have at least `n` elements'
        self.delays = delays[:n, :self.total_rounds] # (workers, rounds + T=B)
        
        # state of the master: (workers, round)
        self.state = np.full((n , self.total_rounds), np.nan)    
        self.durations = np.full((self.total_rounds, ), -1.)
        self.num_waits = 0

    
    @classmethod
    def normalized_load(cls, n, *args):
        if len(args) == 1:
            s = args[0]
        elif len(args) == 3:
            B, W, lambd = args
            s = math.ceil((B * lambd) / (W - 1 + B))
        else: 
            raise ValueError('number of params should be either 1 (s) or 3 (B, W, lambd)')
        return (s + 1) / n
    

    @classmethod
    def param_combinations(cls, n, rounds, max_delay, max_W=None):
        max_W = max_W or rounds
        for lambd in range(1, n+1):
            for B in range(1, max_delay+1):
                for W in range(B+1, max_W+1, B):  # W = x * B + 1
                    yield B, W, lambd

    @classmethod
    def delay(cls, B, W, lambd):
        return B

    
    def run(self) -> None:
        self.num_waits = 0
        for round_ in range(self.total_rounds):
            # perform round
            self.perform_round(round_)
            
            # decode
            job = self.get_decodable_job(round_)
            if job >= 0 and not self.is_decodable(job):
                raise RuntimeError(f'Job {job} in round {round_} is not decodable.')
                 
    
    def perform_round(self, round_) -> None:
        """ This will fill state(:,  round_) """
        
        decode_job = self.get_decodable_job(round_)
        
        # if we are in last B rounds, we might not need to perform
        if (round_ >= self.n_jobs) and self.is_decodable(decode_job):
            round_duration = 0
            round_result = np.full((self.n, ), np.nan)
            
        else:
            # every worker by default returns task for round (round_)
            round_result = np.full((self.n, ), round_)

            # if there are v > s stragglers in round (round_ - B), v-s of
            # those stragglers repeat their (round_ - B) task instead of the 
            # current task
            if decode_job >= 0:
                no_contrib_workers = np.flatnonzero(self.state[:, decode_job] != decode_job)
                if (v := no_contrib_workers.size) > self.s:
                    repeat_workers = no_contrib_workers[0 : v - self.s]
                    round_result[repeat_workers] = decode_job

            # apply stragglers
            delay = self.delays[:, round_]
            wait_time = delay.min() * (1 + self.mu) 
            is_straggler = delay > wait_time

            if self.follows_straggler_model(round_, is_straggler, round_result):
                # do not wait for all: apply straggler pattern
                round_result[is_straggler] = -1
                round_duration = np.minimum(wait_time, delay.max())
            else:
                # wait for all: do not apply stragglers
                self.num_waits += 1
                round_duration = delay.max()

        # set round_result into state
        self.state[:, round_] = round_result
        self.durations[round_] = round_duration


    def is_decodable(self, job) -> bool:
        """
        checks whether a job can be decoded.
        To be able to decode job t, there should be at least n - s tasks received
        from workers in rounds t and t + B.
        """
        task_results = self.task_results(job) # (2 * n, )
        return (task_results == job).sum() >= self.n - self.s
    
        
    def get_decodable_job(self, round_) -> int:
        """ returns the job decodable in round (round_) """
        return round_ - self.B
    
    
    def task_results(self, job) -> np.ndarray:
        """ returns the recieved tasks from workers in rounds job and job+B.
                shape: (2*n,) 
        """

        if job > self.n_jobs:
            raise ValueError('job > rounds')
        return np.concatenate((self.state[:, job],
                               self.state[:, job+self.B]))
    
    
    def follows_straggler_model(self, r, is_straggler, round_result) -> bool:
        """ Checks if at any given round, the spatial and temporal conditions 
            of (B, W, lambd)-bursty straggler model are met.
            
            1- spatial correlation: within the past W rounds, at most `lambd`
            unique stragglers.
            2- temporal correlation: if worker i is a straggler at the current
            round, it cannot be a straggler in [-W, -B] rounds relative to 
            the current round.
            
            r (int): current round idx.
            is_straggler (ndarray): boolean array of length n. Straggler pattern 
                of the current round.
            round_result (ndarray): result of the current round.
        """
        
        # 1. spatial cond: at most `lambd` unique stragglers over the 
        # past W rounds.
        window = np.arange(np.maximum(0, r+1-self.W), r)
        been_straggler = self.stragglers(window, exemption=True)       
        
        num_stragglers = (been_straggler | is_straggler).sum()
        if num_stragglers > self.lambd:
            return False
        
        # 2. temporal cond: if worker i is a straggeler at the 
        # current round, it cannot be a straggeler in [-W, -B]:
        window = np.arange(np.maximum(0, r+1-self.W), np.maximum(0, r+1-self.B))
        been_straggler = self.stragglers(window, exemption=True)       

        if (been_straggler & is_straggler).any():
            return False
        
        return True
    
    
    def stragglers(self, window, exemption=True):
        state_window = self.state[:, window]
        been_straggler = (state_window == -1)

        # exclude some rounds from straggler counts:
        if exemption:
            exempt1 = np.all((state_window == window) | (state_window == -1), axis=0)
            exempt2 = (state_window == -1).sum(axis=0) <= self.s
            exempt = exempt1 & exempt2 
            been_straggler[:, exempt] = False
        
        been_straggler = been_straggler.any(axis=1)
        return been_straggler

        
