''' taken from:
https://github.com/rashisht1/gradient_coding/blob/1ba503508612ad92baa112676735e642668f4fb8/src/util.py
'''

import itertools

import numpy as np
import scipy.special as sp



def get_B(n_workers, n_stragglers):
    Htemp=np.random.normal(0,1,[n_stragglers,n_workers-1])
    H=np.vstack([Htemp.T,-np.sum(Htemp,axis=1)]).T

    Ssets=np.zeros([n_workers,n_stragglers+1])

    for i in range(n_workers):
        Ssets[i,:]=np.arange(i,i+n_stragglers+1)
    Ssets=Ssets.astype(int)
    Ssets=Ssets%n_workers
    B=np.zeros([n_workers,n_workers])
    for i in range(n_workers):
        B[i,Ssets[i,0]]=1
        vtemp=-np.linalg.solve(H[:,np.array(Ssets[i,1:])],H[:,Ssets[i,0]])
        ctr=0
        for j in Ssets[i,1:]:
            B[i,j]=vtemp[ctr]
            ctr+=1

    return B

def get_A(B, n_workers, n_stragglers):
    #S=np.array(list(itertools.permutations(np.hstack([np.zeros(n_stragglers),np.ones(n_workers-n_stragglers)]),n_workers)))
    #print(S)
    #S=unique_rows(S)
    
    S = np.ones((int(sp.binom(n_workers,n_stragglers)),n_workers))
    combs = itertools.combinations(range(n_workers), n_stragglers)
    i=0
    for pos in combs:
        S[i,pos] = 0
        i += 1

    (m,n)=S.shape
    A=np.zeros([m,n])
    for i in range(m):
        sp_pos=S[i,:]==1
        A[i,sp_pos]=np.linalg.lstsq(B[sp_pos,:].T,np.ones(n_workers))[0]

    return A


def find_coef(B, s, is_straggler):
    n = B.shape[0]
    I = np.where(~is_straggler)[0]
    
    if len(I) < n - s:
        return None
    else:
        I = I[:(n-s)]   
        a = np.zeros((n, ))
        a[I] = np.linalg.lstsq(B[I,:].T, np.ones((n, )), rcond=None)[0]
        return a