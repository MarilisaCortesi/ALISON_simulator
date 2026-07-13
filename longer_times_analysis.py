import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def get_cancer(dt):
	out = {}
	for tr in dt:
		time = list(dt[tr][0].keys())
		time.remove('initial_condition')
		time = sorted(time)
		time.insert(0, 'initial_condition')
		out[tr] = {}
		for r in dt[tr]:
			out[tr][r] = []
			for t in time:
				cell_pop = dt[tr][r][t]['cell_population']
				cancer = 0
				for c in cell_pop:
					if c.type =='cancer':
						cancer+=1
				out[tr][r].append(cancer)
	return out

def plot_cancer(cnr, flout):
	for tr in cnr:
		f, ax = plt.subplots()
		time = np.arange(0,100+tr,tr)
		for r in cnr[tr]:
			ax.plot(time, cnr[tr][r], marker='o', color='b')
		ax.set_title('resolution:' + str(tr))
		flout.savefig()

def average_cancer(cnr):
	out = {}
	for tr in cnr:
		out[tr] = {'M':[], 'S':[]}
		time = range(len(cnr[tr][0]))
		for t in time:
			temp = []
			for r in cnr[tr]:
				temp.append(cnr[tr][r][t])
			print(temp)
			out[tr]['M'].append(np.mean(temp))
			out[tr]['S'].append(np.std(temp))
	return out


def plot_average(ave_c, fout):
	colors = {1:'k', 2:'r', 5:'g', 10:'b'}
	f, ax = plt.subplots()
	for tr in ave_c:
		time = np.arange(0,100+tr,tr)
		ax.errorbar(time, ave_c[tr]['M'], yerr=ave_c[tr]['S'], color=colors[tr])
	fout.savefig()


input_folder = '/Users/marilisacortesi/Desktop/longer_time/outputs/'
files = os.listdir(input_folder)
file_out = PdfPages('/Users/marilisacortesi/Desktop/longer_time/PEO1_longer_times_comparison.pdf')

data = {}
for f in files:
	if f.startswith('.'):
		continue
	time = int(f.split('configuration_')[2].split('h')[0])
	if time not in data:
		data[time] = {}
	with open(input_folder+f, 'rb') as F:
		data[time][len(data[time])] = pickle.load(F)

cancer = get_cancer(data)

plot_cancer(cancer, file_out)

ave_cancer = average_cancer(cancer)

plot_average(ave_cancer, file_out)

file_out.close()