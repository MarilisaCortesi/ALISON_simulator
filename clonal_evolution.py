import pickle
import random
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



starting_population=10 # number of cells in the starting population
duration = 365 #days
clones = {}
doubling_rates = {}
metastasis_rates = {}
same_features_rates = {}
treatment_sensitivity = {}
death_rates = {}
for c in conditions:
	clones[c] = {0:starting_population}
	doubling_rates[c] = {0: conditions[c]['doubling_time']}
	metastasis_rates[c] = {0:conditions[c]['metastasis_rate']}
	same_features_rates[c] = {0: conditions[c]['same_features']}
	treatment_sensitivity[c] = {0: conditions[c]['treatment_sensitivity']}
	death_rates[c] = {0: conditions[c]['death_rate']}
	for t in range(duration):
		for cc in range(len(clones[c])):
			if clones[c][cc]<=0:
				clones[c][cc] = 0
				continue
			metastasis = random.random()*100
			if metastasis<metastasis_rates[c][cc]: # new clone
				idx = len(clones[c])
				clones[c][idx] = starting_population #Hp that the new clone starts with the same number of cells as the starting population
				clones[c][cc]-= starting_population
				keep_features = random.random()*100
				if keep_features < same_features_rates[c][cc]: #keep same features of parent clone
					doubling_rates[c][idx] = doubling_rates[c][cc]
					metastasis_rates[c][idx] = metastasis_rates[c][cc]
					same_features_rates[c][idx] = same_features_rates[c][cc]
					death_rates[c][idx] = death_rates[c][cc]
					treatment_sensitivity[c][idx] = treatment_sensitivity[c][cc]
				else:
					doubling_rates[c][idx] = random.randrange(range_doubling_time[0]*100, range_doubling_time[1]*100, 1)/100
					treatment_sensitivity[c][idx] = random.randrange(treatment_sensitivity_range[0], treatment_sensitivity_range[1], 1)
					if treatment_sensitivity[c][idx]>0.5: # somewhat resistant
						same_features_rates[c][idx] = random.randrange(same_features_range[1], same_features_range[2], 1)
						death_rates[c][idx] = random.randrange(death_rate_range[0], death_rate_range[1], 1)
						metastasis_rates[c][idx] = random.randrange(metastasis_rate_range[1], metastasis_rate_range[2], 1)
					else:
						same_features_rates[c][idx] = random.randrange(same_features_range[0], same_features_range[1], 1)
						death_rates[c][idx] = random.randrange(death_rate_range[1], death_rate_range[2], 1)
						metastasis_rates[c][idx] = random.randrange(metastasis_rate_range[0], metastasis_rate_range[1], 1)
			else:
				population_change = doubling_rates[c][cc]*clones[c][cc] -((death_rates[c][cc]/100)*clones[c][cc])
				clones[c][cc]+= population_change
out_var = {'clones': clones, 'doubling_rates': doubling_rates, 'metastasis_rates': metastasis_rates, 'same_features': same_features_rates,
		   'treatment': treatment_sensitivity, 'death_rate': death_rates}
fileout = '/scratch/mcortesi/outputs/clones.pkl'
with open(fileout, 'wb') as F:
	pickle.dump(F, out_var)









