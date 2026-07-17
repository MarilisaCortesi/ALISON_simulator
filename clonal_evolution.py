import pickle
import random

import numpy as np

file_conditions = 'conditions_evolution.pkl'
with open(file_conditions, 'rb') as F:
	conditions = pickle.load(F)
'''
doubling_rate = 1 # fraction of cells that double in 24h
metastasis_rate = 0.1 # probability of creation of a new clone (metastasis)
same_features = 0.5 # probability that the new clone will maintain the same features of their origin
death_rate = 0.01 # rate of death within the culture
treatment_sensitivity = 1 # percentage of sensitivity to treatment
'''
range_doubling_time = [0.83, 5] # 20 to 120 h
metastasis_rate_range = [1, 25, 50]
same_features_range = [1,50, 100]
death_rate_range = [1,25, 50]
treatment_sensitivity_range = [0,1]
max_clones = 1000


starting_population=10 # number of cells in the starting population
duration = 3 #days

list_conds = list(conditions.keys())
for c in list_conds[0:200]:
	clones = {0:{0:starting_population}}
	doubling_rates = {0:{0:conditions[c]['doubling_time']}}
	metastasis_rates = {0:{0:conditions[c]['metastasis_rate']}}
	same_features_rates = {0:{0:conditions[c]['same_features']}}
	treatment_sensitivity = {0:{0:conditions[c]['treatment_sensitivity']}}
	death_rates = {0:{0:conditions[c]['death_rate']}}
	for t in range(duration):
		if len(clones[t])>max_clones:
			break
		clones[t+1]=clones[t].copy()
		doubling_rates[t+1] = doubling_rates[t].copy()
		metastasis_rates[t+1] = metastasis_rates[t].copy()
		same_features_rates[t+1] = same_features_rates[t].copy()
		treatment_sensitivity[t+1] = treatment_sensitivity[t].copy()
		death_rates[t+1]= death_rates[t].copy()
		for cc in range(len(clones[t+1])):
			if clones[t+1][cc]<=0:
				clones[t+1][cc] = 0
				continue
			metastasis = random.random()*100
			if metastasis<metastasis_rates[t+1][cc]: # new clone
				idx = len(clones[t+1])
				clones[t+1][idx] = starting_population #Hp that the new clone starts with the same number of cells as the starting population
				clones[t+1][cc]-= starting_population
				keep_features = random.random()*100
				if keep_features < same_features_rates[t+1][cc]: #keep same features of parent clone
					doubling_rates[t+1][idx] = doubling_rates[t+1][cc]
					metastasis_rates[t+1][idx] = metastasis_rates[t+1][cc]
					same_features_rates[t+1][idx] = same_features_rates[t+1][cc]
					death_rates[t+1][idx] = death_rates[t+1][cc]
					treatment_sensitivity[t+1][idx] = treatment_sensitivity[t+1][cc]
				else:
					doubling_rates[t+1][idx] = random.randrange(range_doubling_time[0]*100, range_doubling_time[1]*100, 1)/100
					treatment_sensitivity[t+1][idx] = random.randrange(treatment_sensitivity_range[0], treatment_sensitivity_range[1], 1)
					if treatment_sensitivity[t+1][idx]>0.5: # somewhat resistant
						same_features_rates[t+1][idx] = random.randrange(same_features_range[1], same_features_range[2], 1)
						death_rates[t+1][idx] = random.randrange(death_rate_range[0], death_rate_range[1], 1)
						metastasis_rates[t+1][idx] = random.randrange(metastasis_rate_range[1], metastasis_rate_range[2], 1)
					else:
						same_features_rates[t+1][idx] = random.randrange(same_features_range[0], same_features_range[1], 1)
						death_rates[t+1][idx] = random.randrange(death_rate_range[1], death_rate_range[2], 1)
						metastasis_rates[t+1][idx] = random.randrange(metastasis_rate_range[0], metastasis_rate_range[1], 1)
			else:
				population_change = doubling_rates[t+1][cc]*clones[t+1][cc] -((death_rates[t+1][cc]/100)*clones[t+1][cc])
				clones[t+1][cc]+= population_change
	out_var = {'clones': clones, 'doubling_rates': doubling_rates, 'metastasis_rates': metastasis_rates, 'same_features': same_features_rates,
		   'treatment': treatment_sensitivity, 'death_rate': death_rates}
	fileout = 'clones_'+str(c)+'.pkl'
	with open(fileout, 'wb') as F:
		pickle.dump(out_var, F)









