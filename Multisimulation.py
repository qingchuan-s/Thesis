#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sciris as sc
import covasim as cv
import optuna as op


# In[2]:


def create_sim():

    beta = 0.0095
    pop_infected =130 # initial cases of infection
   
    start_day = '2020-02-01'
    end_day   = '2020-12-31'
    data_file = 'Denmark.csv'
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
       # rel_death_prob =  rel_death_prob,
        asymp_factor = asymp_factor,
        rescale      = True,
        verbose      = 0.1,
    )
    # Create the baseline simulation
    sim = cv.Sim(pars=pars, datafile=data_file, location='denmark')
    
    # Day 1: 2020-02-01
    # Day 138: 2020-06-17
    # Day 260: 2020-10-17
    relative_death = cv.dynamic_pars(rel_death_prob=dict(days=[1,138,260], vals=[0.93,0.336,0.3]))
    
    ### Change beta ###
    beta_days = ['2020-03-20','2020-04-15','2020-05-10','2020-06-08','2020-06-22','2020-07-28','2020-08-22','2020-08-31','2020-09-17',
                '2020-10-25','2020-11-05','2020-11-20','2020-12-07','2020-12-20','2020-12-25']
    
    h_beta_changes = [1.25, 1.25, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.25, 1.25]
    s_beta_changes = [0.02, 0.10, 0.20, 0.40, 0.05, 0.05, 0.05, 0.80, 0.80, 0.70, 0.65, 0.70, 0.30, 0.02, 0.02]
    w_beta_changes = [0.20, 0.20, 0.30, 0.30, 0.50, 0.50, 0.60, 0.70, 0.70, 0.60, 0.55, 0.60, 0.50, 0.40, 0.10]
    c_beta_changes = [0.20, 0.30, 0.40, 0.60, 0.90, 0.90, 0.80, 0.80, 0.70, 0.60, 0.60, 0.70, 0.70, 0.75, 0.60]
    
    # Define the beta changes
    h_beta = cv.change_beta(days=beta_days, changes=h_beta_changes, layers='h')
    s_beta = cv.change_beta(days=beta_days, changes=s_beta_changes, layers='s')
    w_beta = cv.change_beta(days=beta_days, changes=w_beta_changes, layers='w')
    c_beta = cv.change_beta(days=beta_days, changes=c_beta_changes, layers='c')

    interventions = [h_beta, w_beta, s_beta, c_beta, relative_death ]
    
    # import infections from 2020-02-20 to 2020-03-11
    imports = cv.dynamic_pars(n_imports=dict(days=[20, 40], vals=[1, 0]))
    
    interventions += [imports]
    
    # Time to change the test plan
    day_march = sim.day('2020-03-16') 
    day_april = sim.day('2020-04-01') 
    day_may = sim.day('2020-05-05') 
    day_june = sim.day('2020-06-08')
    day_july = sim.day('2020-07-01')
    day_august = sim.day('2020-08-01')
    day_sep = sim.day('2020-09-07')
    day_october = sim.day('2020-10-01')
    day_nov = sim.day('2020-11-01')
    day_dec = sim.day('2020-12-01')
    
    # testing probability for symptomatic
    s_prob_initial = 0.0075 # suppose 10% of symptimatic people will go to test
    s_prob_march = 0.0158 # suppose 20% of symptimatic people will go to test
    s_prob_april = 0.0251 # with a comprehensive testing plan, suppose 30% of symptomatic people will go to test
    s_prob_may   = 0.0251
    s_prob_june = 0.0251
    s_prob_july = 0.0251
    s_prob_august = 0.0358 # suppose 40% of symptimatic people will go to test
    s_prob_sep = 0.0483 # suppose 50% of symptimatic people will go to test
    s_prob_october = 0.0483
    s_prob_nov = 0.0554 # suppose 55% of symptimatic people will go to test
    s_prob_dec = 0.0722 # suppose 65% of symptimatic people will go to test
    
    t_delay       = 1.0 
    iso_vals = [{k:0.1 for k in 'hswc'}]
    
    # testing, tracing and isolation 
    # From May 12, starting tracing and isolation strategy
    tti_day =  sim.day('2020-05-12')
    tracing_prob = dict(h=1.0, s=0.5, w=0.5, c=0.2)
    trace_time   = {'h':0, 's':1, 'w':1, 'c':2}   
    
    #testing and isolation intervention
    interventions += [
        cv.test_prob(symp_prob=s_prob_initial, asymp_prob=0.0, symp_quar_prob=0.0, asymp_quar_prob=0.0, start_day=start_day, end_day=day_march-1, test_delay=t_delay),
        cv.test_prob(symp_prob=s_prob_march, asymp_prob=0.0, symp_quar_prob=0.0, asymp_quar_prob=0.0, start_day=day_march, end_day=day_april-1, test_delay=t_delay),
        cv.test_prob(symp_prob=s_prob_april, asymp_prob=0.0, symp_quar_prob=0.0, asymp_quar_prob=0.0, start_day=day_april, end_day=day_may-1, test_delay=t_delay),
        cv.test_prob(symp_prob=s_prob_may, asymp_prob=0.00075, symp_quar_prob=0.00075, asymp_quar_prob=0.0, start_day=day_may, end_day=day_june-1, test_delay=t_delay),
        cv.test_prob(symp_prob=s_prob_june, asymp_prob=0.00075, symp_quar_prob=0.00075, asymp_quar_prob=0.0, start_day=day_june, end_day = day_july-1, test_delay=t_delay),
        cv.test_prob(symp_prob=s_prob_july, asymp_prob=0.00075, symp_quar_prob=0.00075, asymp_quar_prob=0.0, start_day=day_july, end_day = day_august-1, test_delay=t_delay),
        cv.test_prob(symp_prob=s_prob_august, asymp_prob=0.00075, symp_quar_prob=0.00075, asymp_quar_prob=0.0, start_day=day_august, end_day = day_sep-1, test_delay=t_delay),
        cv.test_prob(symp_prob=s_prob_sep, asymp_prob=0.001, symp_quar_prob=0.001, asymp_quar_prob=0.0, start_day=day_sep, end_day=day_october-1, test_delay=t_delay),
        cv.test_prob(symp_prob=s_prob_october, asymp_prob=0.001, symp_quar_prob=0.001, asymp_quar_prob=0.0, start_day=day_october, end_day=day_nov-1, test_delay=t_delay),
        cv.test_prob(symp_prob=s_prob_nov, asymp_prob=0.001, symp_quar_prob=0.005, asymp_quar_prob=0.001, start_day=day_nov, end_day=day_dec-1, test_delay=t_delay),
        cv.test_prob(symp_prob=s_prob_dec, asymp_prob=0.0075, symp_quar_prob=0.0203, asymp_quar_prob=0.0075, start_day=day_dec, test_delay=t_delay),
        cv.dynamic_pars({'iso_factor': {'days': day_march, 'vals': iso_vals}}),
        cv.contact_tracing(trace_probs=tracing_prob, trace_time=trace_time, start_day=tti_day),
      ]
    
    sim.update_pars(interventions=interventions)
    
    for intervention in sim['interventions']:
        intervention.do_plot = False
    return sim


# In[ ]:


sim = create_sim()
msim = cv.MultiSim(sim)

msim.run(n_runs=20)

msim.mean()
msim.plot(to_plot=['cum_tests', 'cum_diagnoses', 'cum_deaths'])

