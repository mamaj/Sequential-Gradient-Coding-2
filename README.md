# CODE DOCS

## 1. Probing profile:  

`sam-gc-cnn_profile_est_desktop_long4_new`

workers=256
invokes=100
profile_loads=[0.0, 0.25, 0.5, 0.75, 1.0]
batch=4096
comp_type='no_forloop'
region='Canada'



## 2. Selecting Scheme Parameters 

1. fix max_delay
2. change n_jobs: 20 40 60 80

base_load = 0.0
mu = 1.0

# total number of rounds profiled - number of jobs to complete
n_jobs = 80  # number of jobs to complete
max_delay = invokes - n_jobs  # T = 20



## 3. Experiments

`sam-gc-cnn_profile_est_desktop_long4_new_real_2`

