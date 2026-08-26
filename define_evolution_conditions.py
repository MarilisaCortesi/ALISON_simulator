import random
import scipy.stats.qmc
import numpy as np
import pickle as pkl


doubling_times = np.arange(0.4, 2.4, 0.01) # da correggere
metastasis_rate = np.arange(0,10,0.005)
same_features_range = np.arange(1, 100,0.5)  #correlate with metastasis rate
death_rate_range = np.arange(0,51,0.5)
treatment_sensitivity_range = np.arange(0,100,1) # inversely correlated width death rate

n_conditions = 1000
sampler = scipy.stats.qmc.LatinHypercube(1)
doubling_ids = np.rint(sampler.random(n_conditions)*100)

conditions = {}
for id, d in enumerate(doubling_ids):
	idx = int(d[0])
	conditions[id] = {'doubling_time': doubling_times[idx]}
	conditions[id]['metastasis_rate'] = metastasis_rate[idx]
	conditions[id]['death_rate'] = death_rate_range[idx]
	met_rate_perc = metastasis_rate[idx]/(max(metastasis_rate)- min(metastasis_rate))
	jitter_same_features = random.uniform(-0.01, 0.01)
	same_features_perc = 100 - ((100*met_rate_perc)+jitter_same_features)
	same_features_rate = (same_features_perc/100)*(max(same_features_range)-min(same_features_range))
	conditions[id]['same_features'] = same_features_rate
	death_rate_perc = death_rate_range[idx]/(max(death_rate_range)-min(death_rate_range))
	jitter_treatment = random.uniform(-0.01, 0.01)
	treatment_sensitivity_perc = death_rate_perc+jitter_treatment
	treatment_sensitivity_rate = treatment_sensitivity_perc*(max(treatment_sensitivity_range)-min(treatment_sensitivity_range))
	conditions[id]['treatment_sensitivity']= treatment_sensitivity_rate


file_out = 'conditions_evolution.pkl'
with open(file_out, 'wb') as F:
	pkl.dump(conditions, F)