import pandas as pd
import numpy as np
import sciris as sc
import covasim as cv #python -m pip install covasim==2.1.0
import pylab as pl
from covasim import base as cvb
from covasim import defaults as cvd
from covasim import utils as cvu
from covasim.interventions import get_subtargets, process_daily_data

#%%
class dose_scheduler(cv.Intervention):
    '''
    Scheduler for doses
    To use update scheduler dictionary with key for each day and value list of {'inds', and 'rel_sys', etc}
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Initialize the Intervention object
        self._store_args()  # Store the input arguments so the intervention can be recreated
        self.scheduler = dict()

    def initialize(self, sim):
        self.initialized = True
        return

    def apply(self, sim):
        if sim.t in self.scheduler.keys():
            for schedule in self.scheduler[sim.t]:
                # pop out inds
                vacc_inds = schedule.pop('inds', np.array([], cvd.default_int))
                # look through rel factors
                for k, v in schedule.items():
                    sim.people[k][vacc_inds] = v
            # clean up
            self.scheduler.pop(sim.t)
        return
    
#%%
class vaccine_plan(cv.Intervention):
    def __init__(self,  daily_vaccines, start_day=None, end_day=None, delay=None, prob=1.0, rel_symp=None, rel_sus=None,
                 subtarget=None, cumulative=[0.5, 1], **kwargs):
        super().__init__(**kwargs) # NB: This line must be included
        self._store_args()  # Store the input arguments so the intervention can be recreated
        self.start_day   = start_day
        self.end_day     = end_day
        self.delay =  cvd.default_int(delay) # days needed to take the second dose
        self.rel_sus     = rel_sus  # relative change in susceptibility; 0 = perfect, 1 = no effect
        self.rel_symp = rel_symp # relative change in symptom probability for people who still get infected; 0 = perfect, 1 = no effect
        self.daily_vaccines =  daily_vaccines #daily number of vaccines (first dose)
        self.subtarget = subtarget
        if cumulative in [0, False]:
            cumulative = [1,0] # First dose has full efficacy, second has none
        elif cumulative in [1, True]:
            cumulative = [1] # All doses have full efficacy
        self.cumulative = np.array(cumulative, dtype=cvd.default_float) # Ensure it's an array
        return


    def initialize(self, sim):
        ''' Fix the dates and store the vaccinations '''
        # Handle days
        self.start_day   = sim.day(self.start_day)
        self.end_day     = sim.day(self.end_day)
        self.days        = [self.start_day, self.end_day]
        
        # Process daily data
        self.daily_vaccines = process_daily_data(self.daily_vaccines, sim, self.start_day)

        # Ensure we have the dose scheduler
        flag = True
        for intv in sim['interventions']:
            if isinstance(intv, dose_scheduler):
                flag = False
        if flag:
            sim['interventions'] += [dose_scheduler()]
        
        # Save
        self.orig_rel_sus      = sc.dcp(sim.people.rel_sus) # Keep a copy of pre-vaccination susceptibility
        self.orig_symp_prob    = sc.dcp(sim.people.symp_prob) # ...and symptom probability
        #self.mod_rel_sus       = np.ones(sim.n, dtype=cvd.default_float) # Store the final modifiers
        #self.mod_symp_prob     = np.ones(sim.n, dtype=cvd.default_float) # Store the final modifiers

            
        # Initialize vaccine info
        self.vaccinations = np.zeros(sim.n, dtype=cvd.default_int)
        #self.vaccine_take = np.zeros(sim.n, dtype=np.bool)
        self.delay_days = np.zeros(sim.n, dtype=cvd.default_int)
        self.first_dates = np.zeros(sim.n, dtype=cvd.default_int)  # Store the dates when people are vaccinated the first time
        sim.results['new_doses'] = cvb.Result(name='New Doses', npts=sim['n_days']+1, color='#ff00ff')
        self.initialized = True

        return

    def apply(self, sim):
        ''' Perform vaccination '''

        t = sim.t
        if t <   self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        # Check that there are still vaccines
        rel_t = t - self.start_day
        if rel_t < len(self.daily_vaccines):
            n_vaccines = sc.randround(self.daily_vaccines[rel_t]/sim.rescale_vec[t]) 
        else:
            return
                
        vacc_probs = np.ones(sim.n) # Begin by assigning equal vaccine weight (converted to a probability) to everyone
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
            vacc_probs[subtarget_inds] = subtarget_vals # People being explicitly subtargeted
        
        # First dose
        # Don't give dose to people who have had at least one dose
        vacc_inds = cvu.true(self.vaccinations > 0)
        vacc_probs[vacc_inds] = 0.0

        # Now choose who gets vaccinated and vaccinate them
        n_vaccines = min(n_vaccines, (vacc_probs!=0).sum()) 
        all_vacc_inds = cvu.choose_w(probs=vacc_probs, n=n_vaccines, unique=True) # Choose who actually tests
        sim.results['new_doses'][t] += len(all_vacc_inds)
        
        self.vaccinations[all_vacc_inds] = 1
        self.delay_days[all_vacc_inds] = self.delay
        self.first_dates[all_vacc_inds] = sim.t
        
        # Calculate the effect per person
        vacc_eff = self.cumulative[0]  # Pull out corresponding effect sizes
        rel_sus_eff = 0.5+0.5*self.rel_sus
        rel_symp_eff = 0.5+0.5*self.rel_symp           
        # Apply the vaccine to people
        #sim.people.rel_sus[all_vacc_inds] *= rel_sus_eff
        #sim.people.symp_prob[all_vacc_inds] *= rel_symp_eff
        sim.people.rel_sus[all_vacc_inds] = self.orig_rel_sus[all_vacc_inds]*rel_sus_eff
        sim.people.symp_prob[all_vacc_inds] = self.orig_symp_prob[all_vacc_inds]*rel_symp_eff
        
        all_vacc_inds2 =  cvu.true((self.vaccinations == 1)*(self.delay_days >0)*(self.first_dates >0)*(self.first_dates + self.delay_days == sim.t))
        rel_sus_eff2 =  self.rel_sus
        rel_symp_eff2 =  self.rel_symp
        #sim.people.rel_sus[all_vacc_inds2] *= rel_sus_eff2
        #sim.people.symp_prob[all_vacc_inds2] *= rel_symp_eff2
        sim.people.rel_sus[all_vacc_inds2] = self.orig_rel_sus[all_vacc_inds2]*rel_sus_eff2
        sim.people.symp_prob[all_vacc_inds2] = self.orig_symp_prob[all_vacc_inds2]*rel_symp_eff2
        self.vaccinations[all_vacc_inds2] = 2
        sim.results['new_doses'][t] += len(all_vacc_inds2)

#%%
# Define the test subtargeting
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
# Define the vaccine subtargeting
def vaccinate_by_age(sim):
    # cv.true() returns indices of people matching this condition, i.e. people under 50
    young_or_sick  = cv.true((sim.people.age < 16)+(sim.people.susceptible == False)) 
    #middle_danger = cv.true((sim.people.age >= 16) * (sim.people.age < 65) * (sim.people.exposed == True))
    old  = cv.true((sim.people.age >= 65)*(sim.people.susceptible == True))
    others =  cv.true((sim.people.age >= 16) * (sim.people.age < 65)  * (sim.people.susceptible == True))
    inds = sim.people.uid # Everyone in the population -- equivalent to np.arange(len(sim.people))
    vals = np.ones(len(sim.people)) # Create the array
    vals[young_or_sick] = 0 
    #vals[middle_danger] = 5 
    vals[old] = 10 
    vals[others] = 0.1
    output = dict(inds=inds, vals=vals)
    return output

#%%
# read data
df = pd.read_csv('E:/thesis/data/Denmark_data.csv')
#df =df[0:275] # till 2020-11-30
df =df[0:396] # till 2021-03-31

daily = pd.read_csv('E:/thesis/data/daily vaccine.csv',encoding='utf-8')
daily = daily['Daily dose'].values.tolist()
daily1 = daily[0:9]
daily2 = daily[9:41]
daily3 = daily[41:51]
daily4 = daily[51:85]


vn = []
vn[:] = [25000 for i in range(91)] # Scenario 1 ~ 3
# vn[:] = [40000 for i in range(91)]  # Scenario 4 ~ 6

t = 200000 # Scenario 1,4
# t = 300000 # Scenario 2,5
# t = 400000 # Scenario 3,7

#%%
def create_sim(seed):
    beta = 0.014 #0.011
    pop_infected =10 # initial cases of infection
    start_day = '2020-02-01'
    #end_day   = '2020-11-30'
    #end_day   = '2021-03-31'
    end_day   = '2021-06-30'
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
    sim = cv.Sim(pars=pars, datafile=data_file, location='denmark',analyzers=cv.age_histogram(edges = [0,16,65,130],states=['exposed','symptomatic','dead'])) #edges = [0,16,65,130],
    
    relative_death = cv.dynamic_pars(rel_death_prob=dict(days=[sim.day('2020-02-01'),sim.day('2020-06-09'),sim.day('2020-09-17')], 
                                                               vals=[1.3,0.8,0.4]))
    interventions = [relative_death]
    
    ### Change beta ###
    beta_days = ['2020-03-15','2020-04-15','2020-05-10','2020-06-22','2020-07-20','2020-08-22','2020-09-01','2020-09-22','2020-10-01','2020-10-15','2020-11-01',
                 '2020-11-20','2020-11-30','2020-12-14', '2021-01-01','2021-03-01','2021-04-06']
    h_beta_changes = [1.10, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.10, 1.10, 1.10, 1.20, 1.30, 1.20, 1.10, 1.00, 1.00]
    s_beta_changes = [0.80, 0.50, 0.50, 0.40, 0.60, 0.60, 1.00, 0.80, 0.80, 1.00, 0.80, 1.00, 1.20, 1.00, 0.80, 0.70, 0.60]
    w_beta_changes = [0.80, 0.50, 0.50, 0.40, 0.60, 0.60, 1.00, 0.80, 0.80, 1.00, 0.80, 1.00, 1.20, 1.00, 0.80, 0.70, 0.50]
    c_beta_changes = [0.90, 0.60, 0.50, 0.70, 0.90, 0.80, 1.00, 0.70, 0.70, 1.00, 0.80, 1.10, 1.30, 1.10, 1.00, 0.80, 0.70]
    
    # Define the beta changes
    h_beta = cv.change_beta(days=beta_days, changes=h_beta_changes, layers='h')
    s_beta = cv.change_beta(days=beta_days, changes=s_beta_changes, layers='s')
    w_beta = cv.change_beta(days=beta_days, changes=w_beta_changes, layers='w')
    c_beta = cv.change_beta(days=beta_days, changes=c_beta_changes, layers='c')
    
    ### edge clipping ###
    clip_days = ['2020-03-15','2020-04-15','2020-05-10','2020-06-08','2020-06-22','2020-08-17','2020-09-01','2020-09-15','2020-10-05',
                 '2020-11-05','2020-11-20','2020-12-09','2020-12-19','2020-12-25','2021-01-04','2021-02-01', '2021-03-01','2021-04-06']
    s_clip_changes = [0.01, 0.20, 0.40, 0.70, 0.05, 0.10, 0.90, 0.80, 0.70, 0.70, 0.70, 0.40, 0.05, 0.05, 0.05, 0.30, 0.50, 0.80]
    w_clip_changes = [0.10, 0.30, 0.50, 0.70, 0.60, 0.80, 1.00, 0.80, 0.70, 0.70, 0.70, 0.60, 0.40, 0.10, 0.50, 0.60, 0.60, 0.80]
    c_clip_changes = [0.20, 0.40, 0.60, 0.85, 1.00, 1.00, 1.00, 0.80, 0.80, 0.90, 0.90, 0.70, 0.80, 0.50, 0.60, 0.70, 0.80, 0.90]
    
    # Define the edge clipping
    s_clip = cv.clip_edges(days=clip_days, changes=s_clip_changes, layers='s')
    w_clip = cv.clip_edges(days=clip_days, changes=w_clip_changes, layers='w')
    c_clip = cv.clip_edges(days=clip_days, changes=c_clip_changes, layers='c')
    interventions += [h_beta, w_beta, s_beta, c_beta, w_clip, s_clip, c_clip]
    
    # Add a new change in beta to represent the takeover of the new variant B.1.1.7
    nv_days   = np.linspace(sim.day('2020-12-14'), sim.day('2021-03-28'), 15*7)
    nv_prop   = 0.952/(1+np.exp(-0.099*(nv_days-sim.day('2020-12-14')-59))) 
    nv_change = nv_prop*1.5 + (1-nv_prop)*1.0 #r = 1.5
    nv_beta = cv.change_beta(days=nv_days, changes=nv_change)
    
    c = np.r_[0.8*np.ones(sim.day('2021-02-13')-sim.day('2020-12-14')), 0.4*np.ones(sim.day('2021-03-29')-sim.day('2021-02-13'))]
    relative_severe = cv.dynamic_pars(rel_severe_prob=dict(days=nv_days, vals=nv_prop*1.2 + (1-nv_prop)*1))
    relative_critical = cv.dynamic_pars(rel_crit_prob=dict(days=nv_days, vals=nv_prop*1.2 + (1-nv_prop)*1))
    relative_death_nv = cv.dynamic_pars(rel_death_prob=dict(days=nv_days, vals=nv_prop*c*1.2+ (1-nv_prop)*c))  
    interventions += [nv_beta,relative_severe,relative_critical,relative_death_nv]
    
    # import infections from 2020-02-20 to 2020-03-01
    imports1 = cv.dynamic_pars(n_imports=dict(days=[25, 35], vals=[2,0]))
    imports2 = cv.dynamic_pars(n_imports=dict(days=[171, 190], vals=[2,0]))
    interventions += [imports1,imports2]
    
    iso_vals   = [{'h': 0.5, 's': 0.05, 'w': 0.05, 'c': 0.1}] #dict(h=0.5, s=0.05, w=0.05, c=0.1)
    interventions += [cv.dynamic_pars({'iso_factor': {'days': sim.day('2020-03-15'), 'vals': iso_vals }})]
    
    iso_vals2   = [{'h': 0.7, 's': 0.1, 'w': 0.1, 'c': 0.3}] 
    interventions += [cv.dynamic_pars({'iso_factor': {'days': sim.day('2021-05-01'), 'vals': iso_vals2 }})]
    
    # From May 12, starting tracing and isolation strategy
    tracing_prob = dict(h=1.0, s=0.5, w=0.5, c=0.2)
    trace_time   = {'h':0, 's':1, 'w':1, 'c':2}   
    interventions += [cv.contact_tracing(trace_probs=tracing_prob, trace_time=trace_time, start_day='2020-05-12')]
    
    interventions += [cv.test_num(daily_tests=sim.data['new_tests'], start_day=0, end_day=sim.day('2021-03-31'), test_delay=1, symp_test=50,
                    sensitivity=0.97,subtarget= prior_test)]
   
    vaccine1 = vaccine_plan(daily1, start_day='2021-01-06',end_day = '2021-01-24', delay = 22, rel_symp=0.5, rel_sus=0.2,subtarget= vaccinate_by_age) 
    vaccine2 = vaccine_plan(daily2, start_day='2021-01-25',end_day = '2021-02-15', delay = 27, rel_symp=0.5, rel_sus=0.2,subtarget= vaccinate_by_age)
    vaccine3 = vaccine_plan(daily3, start_day='2021-02-16',end_day = '2021-03-07', delay = 24, rel_symp=0.5, rel_sus=0.2,subtarget= vaccinate_by_age)
    vaccine4 = vaccine_plan(daily4, start_day='2021-03-08',end_day = '2021-04-10', delay = 37, rel_symp=0.5, rel_sus=0.2,subtarget= vaccinate_by_age)
    
    
    vaccine5 = vaccine_plan(vn, start_day='2021-04-11',end_day = '2021-06-30', delay = 37, rel_symp=0.5, rel_sus=0.2,subtarget= vaccinate_by_age)
    interventions += [cv.test_num(daily_tests= t, start_day=sim.day('2021-04-01'), end_day=sim.day('2021-06-30'), test_delay=1, symp_test=50,
                    sensitivity=0.97,subtarget= prior_test)]  
      
    interventions += [vaccine1,vaccine2,vaccine3,vaccine4,vaccine5]    
    
    sim.update_pars(interventions=interventions)
    
    for intervention in sim['interventions']:
        intervention.do_plot = False
        
    sim.initialize()
    
    return sim

#%%
sims =[]
for seed in [2,3,4,6,8,9,10,16,20,25,27,33,37,39,43,44,50,58]:
    print(str('Sim')+ str(seed))
    sim =  create_sim(seed=seed)
    sim.run()
    sims.append(sim)














