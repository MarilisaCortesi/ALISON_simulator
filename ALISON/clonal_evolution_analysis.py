import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def compute_n_clones(dt):
	out = {}
	out2 = []
	out3 = []
	for d in dt:
		out[d] = []
		out2.append(len(dt[d]['clones']))
		for t in range(len(dt[d]['clones'])):
			out[d].append(len(dt[d]['clones'][t]))
		out3.append(out[d][-1])
	return out, out2, out3

def plot_n_clones(ncl, pdf):
	f, ax = plt.subplots()
	times = divide_by_time(ncl)
	times_to_plot = np.arange(0,360,step=10)
	temp = []
	for t in times_to_plot:
		temp.append(times[t])
	ax.boxplot(temp)
	pdf.savefig()
	'''
	for i in range(len(times_to_plot)):
		y = temp[i]
		# Add some random "jitter" to the x-axis
		x = np.random.normal(i, 0.04, size=len(y))
		ax.plot(x, y, 'r.', alpha=0.2)
	#ax.set_yscale('symlog')
	'''
	f, ax = plt.subplots()
	for n in ncl:
		ax.plot(len(ncl[n]), ncl[n][-1], marker='.')
	ax.set_yscale('symlog')
	pdf.savefig()

def divide_by_time(ncl):
	out = []
	max_time = 366
	for t in range(max_time):
		temp = []
		for n in ncl:
			try:
				temp.append(ncl[n][t])
			except:
				print('')
				#temp.append(ncl[n][-1])
		out.append(temp)
	return out


folder_data = '/Users/marilisacortesi/Desktop/clonal_evolution/outputs/'
file_out = PdfPages('/Users/marilisacortesi/Desktop/clonal_evolution/plots.pdf')
list_files = os.listdir(folder_data)

data = {}
for f in list_files:
	if f.startswith('.'):
		continue
	clone_id = int(f.split('_')[1].split('.')[0])
	with open(folder_data+f, 'rb') as F:
		data[clone_id] = pkl.load(F)

n_clones, len_sims, max_clones= compute_n_clones(data)
plot_n_clones(n_clones, file_out)
f, ax = plt.subplots()
ax.plot(len_sims,max_clones, marker='.', linestyle='none')
ax.set_xscale('log')
file_out.savefig()
file_out.close()