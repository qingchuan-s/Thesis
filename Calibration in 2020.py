# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:55:43 2021

@author: qingc
"""

import pandas as pd
import numpy as np
import pylab as pl
import sciris as sc
import covasim as cv
import optuna as op

def create_sim(x):
    
    # optimized values
    beta = x[0]
    pop_infected = 130
 

    start_day = '2020-02-01'
    end_day   = '2020-12-31'
    data_file = 'Denmark.csv'
   
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
        #rel_death_prob =  rel_death_prob,
        asymp_factor = asymp_factor,
        rescale      = True,
        verbose      = 0.1,
    )
    
    
    # Create the baseline simulation
    sim = cv.Sim(pars=pars, datafile=data_file, location='denmark')

    # Day 1: 2020-02-01
    # Day 138: 2020-06-17
    # Day 260: 2020-10-17
    relative_death = cv.dynamic_pars(rel_death_prob=dict(days=[1,138,260], vals=[x[1],x[2],x[3]]))
    
    ### Change beta ###
    
    beta_days = ['2020-03-20','2020-04-15','2020-05-10','2020-06-08','2020-06-22','2020-07-28','2020-08-22','2020-08-31','2020-09-17',
                '2020-10-25','2020-11-05','2020-11-20','2020-12-07','2020-12-19','2020-12-25']
    h_beta_changes = [1.25, 1.25, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.25, 1.25]
    s_beta_changes = [0.02, 0.10, 0.20, 0.40, 0.05, 0.05, 0.05, 0.80, 0.80, 0.65, 0.60, 0.65, 0.20, 0.02, 0.02]
    w_beta_changes = [0.20, 0.20, 0.30, 0.30, 0.50, 0.50, 0.60, 0.70, 0.70, 0.60, 0.55, 0.60, 0.50, 0.40, 0.10]
    c_beta_changes = [0.20, 0.30, 0.40, 0.60, 0.90, 0.90, 0.80, 0.80, 0.70, 0.60, 0.55, 0.60, 0.40, 0.50, 0.45]

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
    #day_january21 = sim.day('2021-01-01')


    # testing probability for symptomatic
    s_prob_initial = 0.0075
    s_prob_march = 0.0158 # suppose 20% of symptimatic people will go to test
    s_prob_april = 0.0251 # with a comprehensive testing plan, suppose 30% of symptomatic people will go to test
    s_prob_may   = 0.0251
    s_prob_june = 0.0251
    s_prob_july = 0.0251
    s_prob_august = 0.0358 # suppose 40% of symptimatic people will go to test
    s_prob_sep = 0.0483 # suppose 50% of symptimatic people will go to test
    s_prob_october = 0.0483
    s_prob_nov = 0.0554 # suppose 55% of symptimatic people will go to test
    s_prob_dec = 0.0722 # suppose 60% of symptimatic people will go to test
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
        cv.test_prob(symp_prob=s_prob_dec, asymp_prob=0.005, symp_quar_prob=0.016, asymp_quar_prob=0.005, start_day=day_dec, test_delay=t_delay),
        cv.dynamic_pars({'iso_factor': {'days': day_march, 'vals': iso_vals}}),
        cv.contact_tracing(trace_probs=tracing_prob, trace_time=trace_time, start_day=tti_day),
      ]
    
    sim.update_pars(interventions=interventions)
    for intervention in sim['interventions']:
        intervention.do_plot = False

    return sim

def objective(x):
    ''' Define the objective function we are trying to minimize '''
    # Create and run the sim
    sim = create_sim(x)
    sim.run()
    fit = sim.compute_fit()

    return fit.mismatch


def get_bounds():
    ''' Set parameter starting points and bounds '''
    pdict = sc.objdict(
        beta = dict(best=0.01, lb=0.007, ub=0.012),
        rel_death_prob = dict(best=1.0, lb=0.8, ub=1.2),
        rel_death_prob2 = dict(best=0.7, lb=0.2, ub=1.2),
        rel_death_prob3 = dict(best=0.5, lb=0.2, ub=1.2),
        )

    # Convert from dicts to arrays
    pars = sc.objdict()
    for key in ['best', 'lb', 'ub']:
        pars[key] = np.array([v[key] for v in pdict.values()])

    return pars, pdict.keys()


#Calibration

name      = 'covasim_dk_calibration'
storage   = f'sqlite:///{name}.db'
n_trials  = 200
n_workers = 5

pars, pkeys = get_bounds() # Get parameter guesses


def op_objective(trial):

    pars, pkeys = get_bounds() # Get parameter guesses
    x = np.zeros(len(pkeys))
    for k,key in enumerate(pkeys):
        x[k] = trial.suggest_uniform(key, pars.lb[k], pars.ub[k])

    return objective(x)


def worker():
    study = op.load_study(storage=storage, study_name=name)
    return study.optimize(op_objective, n_trials=n_trials)


def run_workers():
    return sc.parallelize(worker, n_workers)


def make_study():
    try: op.delete_study(storage=storage, study_name=name)
    except: pass
    return op.create_study(storage=storage, study_name=name)


def calibrate():
    ''' Perform the calibration '''
    make_study()
    run_workers()
    study = op.load_study(storage=storage, study_name=name)
    output = study.best_params
    return output, study


def savejson(study):
    dbname = 'calibrated_parameters_DK'

    sc.heading('Making results structure...')
    results = []
    failed_trials = []
    for trial in study.trials:
        data = {'index':trial.number, 'mismatch': trial.value}
        for key,val in trial.params.items():
            data[key] = val
        if data['mismatch'] is None:
            failed_trials.append(data['index'])
        else:
            results.append(data)
    print(f'Processed {len(study.trials)} trials; {len(failed_trials)} failed')

    sc.heading('Making data structure...')
    keys = ['index', 'mismatch'] + pkeys
    data = sc.objdict().make(keys=keys, vals=[])
    for i,r in enumerate(results):
        for key in keys:
            data[key].append(r[key])
    df = pd.DataFrame.from_dict(data)

    order = np.argsort(df['mismatch'])
    json = []
    for o in order:
        row = df.iloc[o,:].to_dict()
        rowdict = dict(index=row.pop('index'), mismatch=row.pop('mismatch'), pars={})
        for key,val in row.items():
            rowdict['pars'][key] = val
        json.append(rowdict)
    sc.savejson(f'{dbname}.json', json, indent=2)

    return

if __name__ == '__main__':

    do_save = True

    to_plot = ['cum_infections', 'new_infections', 'cum_tests', 'new_tests', 'cum_diagnoses', 'new_diagnoses', 'cum_deaths', 'new_deaths']

    # # Plot initial
    print('Running initial...')
    pars, pkeys = get_bounds() # Get parameter guesses
    sim = create_sim(pars.best)
    sim.run()
    sim.plot(to_plot=to_plot)
    objective(pars.best)
    pl.pause(1.0) # Ensure it has time to render

    # Calibrate
    print('Starting calibration for {state}...')
    T = sc.tic()
    pars_calib, study = calibrate()
    sc.toc(T)

    # Plot result
    print('Plotting result...')
    print(str(pars_calib['beta'])+','+ str(pars_calib['rel_death_prob'])+ ','+str(pars_calib['rel_death_prob2'])+ ','+str(pars_calib['rel_death_prob3']))
    sim = create_sim([pars_calib['beta'],pars_calib['rel_death_prob'], pars_calib['rel_death_prob2'],pars_calib['rel_death_prob3']])
    sim.run()
    sim.plot(to_plot=to_plot)
    fit = sim.compute_fit()
    print(fit.mismatches)
    print(fit.mismatch)

    if do_save:
        savejson(study)


print('Done.')