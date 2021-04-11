#%%
import pandas as pd
import numpy as np
import pylab as pl
import sciris as sc
import covasim as cv
import optuna as op
import os

#%%
# Define the vaccine subtargeting
def prior_test(sim):
    old_or_sick  = cv.true((sim.people.age > 60) + (sim.people.symptomatic == True)) 
    others  = cv.true((sim.people.age <= 60) * (sim.people.symptomatic == False))
    inds = sim.people.uid # Everyone in the population -- equivalent to np.arange(len(sim.people))
    vals = np.ones(len(sim.people)) # Create the array
    vals[old_or_sick] = 2
    vals[others] = 0.5
    output = dict(inds=inds, vals=vals)
    return output

#%%
df = pd.read_csv('Denmark_data.csv')
df =df[0:276]

#%%
def create_sim(beta,n0,rd1,rd2,rd3,seed):
    beta = beta
    pop_infected = n0 # initial cases of infection
    start_day = '2020-02-01'
    end_day   = '2020-11-30'
    data_file = df
    
    # Set the parameters
    total_pop    = 5.8e6 # Denmark population size
    pop_size     = 100e3 # Actual simulated population
    pop_scale    = int(total_pop/pop_size)
    pop_type     = 'hybrid'
    asymp_factor = 2
    pars = sc.objdict(
        pop_size     = pop_size,
        pop_infected = pop_infected,
        pop_scale    = pop_scale,
        pop_type     = pop_type,
        start_day    = start_day,
        end_day      = end_day,
        beta         = beta,
        asymp_factor = asymp_factor,
        rescale      = True,
        verbose      = 0.1,
        rand_seed   = seed,
    )
    
    # Create the baseline simulation
    sim = cv.Sim(pars=pars, datafile=data_file, location='denmark')
    
   
    relative_death = cv.dynamic_pars(rel_death_prob=dict(days=[1,130,230], vals=[rd1,rd2,rd3]))
    interventions = [relative_death]
    
    ### beta changes ###
    beta_days = ['2020-03-15','2020-04-15','2020-05-10','2020-06-22','2020-07-20','2020-08-22','2020-09-01','2020-09-22','2020-10-01','2020-10-15','2020-11-01',
                 '2020-11-20']
    h_beta_changes = [1.10, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.10, 1.10, 1.10, 1.20]
    s_beta_changes = [0.80, 0.50, 0.50, 0.40, 0.60, 0.60, 1.00, 0.80, 0.80, 1.00, 0.80, 0.90]
    w_beta_changes = [0.80, 0.50, 0.50, 0.40, 0.60, 0.60, 1.00, 0.80, 0.80, 1.00, 0.80, 0.90]
    c_beta_changes = [0.90, 0.60, 0.50, 0.70, 0.90, 0.80, 1.00, 0.70, 0.70, 1.00, 0.80, 1.10]
    # Define the beta changes
    h_beta = cv.change_beta(days=beta_days, changes=h_beta_changes, layers='h')
    s_beta = cv.change_beta(days=beta_days, changes=s_beta_changes, layers='s')
    w_beta = cv.change_beta(days=beta_days, changes=w_beta_changes, layers='w')
    c_beta = cv.change_beta(days=beta_days, changes=c_beta_changes, layers='c')
    
    ### edge clipping ###
    clip_days = ['2020-03-15','2020-04-15','2020-05-10','2020-06-08','2020-06-22','2020-08-17','2020-09-01','2020-09-15','2020-10-05',
                 '2020-11-05','2020-11-20','2020-12-07','2020-12-19','2020-12-25']
    s_clip_changes = [0.01, 0.20, 0.40, 0.70, 0.05, 0.10, 0.90, 0.80, 0.70, 0.70, 0.70, 0.60, 0.15, 0.05]
    w_clip_changes = [0.10, 0.30, 0.50, 0.70, 0.60, 0.80, 1.00, 0.80, 0.70, 0.70, 0.70, 0.70, 0.70, 0.10]
    c_clip_changes = [0.20, 0.40, 0.60, 0.85, 1.00, 1.00, 1.00, 0.80, 0.80, 0.90, 0.90, 0.90, 1.00, 0.30]
    # Define the edge clipping
    s_clip = cv.clip_edges(days=clip_days, changes=s_clip_changes, layers='s')
    w_clip = cv.clip_edges(days=clip_days, changes=w_clip_changes, layers='w')
    c_clip = cv.clip_edges(days=clip_days, changes=c_clip_changes, layers='c')
    
    interventions += [h_beta, w_beta, s_beta, c_beta, w_clip, s_clip, c_clip]
    
    # import infections from 2020-02-20 to 2020-03-01
    imports1 = cv.dynamic_pars(n_imports=dict(days=[25, 35], vals=[2,0]))
    imports2 = cv.dynamic_pars(n_imports=dict(days=[171, 190], vals=[2,0]))
    interventions += [imports1,imports2]
    
    iso_vals   = [{'h': 0.5, 's': 0.05, 'w': 0.05, 'c': 0.1}] #dict(h=0.5, s=0.05, w=0.05, c=0.1)
    interventions += [cv.dynamic_pars({'iso_factor': {'days': sim.day('2020-03-15'), 'vals': iso_vals }})]
    
    # From May 12, starting tracing and isolation strategy
    tracing_prob = dict(h=1.0, s=0.5, w=0.5, c=0.2)
    trace_time   = {'h':0, 's':1, 'w':1, 'c':2}   
    interventions += [cv.contact_tracing(trace_probs=tracing_prob, trace_time=trace_time, start_day='2020-05-12')]
    
    interventions += [cv.test_num(daily_tests=sim.data['new_tests'], start_day=0, end_day=sim.day(end_day), test_delay=1, symp_test=50,
                    sensitivity=0.97,subtarget= prior_test)]
    
    sim.update_pars(interventions=interventions)
    
    for intervention in sim['interventions']:
        intervention.do_plot = False
    
    sim.initialize()
    
    return sim

#%%

if __name__ == '__main__':

    x = sc.objdict(
        beta = 0.016,
        rd1 = 1.0,) 
        rd2 = 0.8,
        rd3 = 0.6,)
    
    betas = [i / 1000 for i in range(10, 20, 1)]
    n0s = [i  for i in range(10, 100, 10)]
    rd1s = [i / 10 for i in range(8, 15, 1)]
    rd2s = [i / 10 for i in range(5, 10, 1)]
    rd3s = [i / 10 for i in range(3, 7, 1)]
    
    fitting =[[],[],[],[],[],[]]
    for beta in betas:
        for n0 in n0s:
            for rd1 in rd1s:
                for rd2 in rd2s:
                     for rd3 in rd3s:
                         seed =1
                         s0 = create_sim(beta,n0,rd1,rd2,rd3,seed)
                         sims = []
                         for seed in range(21):
                             sim = s0.copy()
                             sim['rand_seed'] = seed
                             sim.set_seed()
                             sim.label = f"Sim {seed}"
                             sims.append(sim)
                         msim = cv.MultiSim(sims)
                         msim.run()
                         for i in range(21):
                             if msim.sims[i].compute_fit().mismatch < 400:
                                 fitting[0].append(beta)
                                 fitting[1].append(n0)
                                 fitting[2].append(rd1)
                                 fitting[3].append(rd2)
                                 fitting[4].append(rd3)
                                 fitting[5].append(msim.sims[i].compute_fit().mismatch)
                        
