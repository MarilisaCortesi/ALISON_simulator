import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from dask.array import corrcoef
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as sts
from matplotlib.lines import lineStyles
from sphinx.directives.code import dedent_lines
from sympy.abc import Q
from sympy.printing.tree import print_node


def divide_by_clones(dt):
	out = {}
	for st in dt:
		for r in dt[st]:
			M_clones = len(dt[st][r]['clones'][365])
			for m in range(M_clones):
				id_clone = len(out)
				out[id_clone] = {'population_size':[], 'doubling_rate':[], 'metastasis_rate': [],
								 'same_features': [], 'treatment':[], 'death_rate': [], 'n_clones':[]}
				start_time = -1
				for t in dt[st][r]['clones']:
					if m in dt[st][r]['clones'][t]:
						start_time = t
						break
				for t2 in range(start_time, 366):
					out[id_clone]['population_size'].append(dt[st][r]['clones'][t2][m])
					out[id_clone]['doubling_rate'].append(dt[st][r]['doubling_rates'][t2][m])
					out[id_clone]['metastasis_rate'].append(dt[st][r]['metastasis_rates'][t2][m])
					out[id_clone]['same_features'].append(dt[st][r]['same_features'][t2][m])
					out[id_clone]['treatment'].append(dt[st][r]['treatment'][t2][m])
					out[id_clone]['death_rate'].append(dt[st][r]['death_rate'][t2][m])
					out[id_clone]['n_clones'].append(M_clones)
	return out





def compute_n_clones(dt):
	out = {}
	out2 = {}
	out3 = {}
	for sp in dt:
		out[sp] = {}
		out2[sp] = []
		out3[sp] = []
		for d in dt[sp]:
			out[sp][d] = []
			out2[sp].append(len(dt[sp][d]['clones']))
			for t in range(len(dt[sp][d]['clones'])):
				out[sp][d].append(len(dt[sp][d]['clones'][t]))
			out3[sp].append(out[sp][d][-1])
	return out, out2, out3

def plot_n_clones(ncl, pdf):
	colors = {10:'b', 100:'g', 1000:'r', 10000:'m', 100000:'k'}
	for n in ncl:
		f, ax = plt.subplots()
		times = divide_by_time(ncl[n])
		times_to_plot = np.arange(0,366,step=10)
		temp = []
		for t in times_to_plot:
			temp.append(times[t])
		ax.boxplot(temp)
		ax.set_title('starting population: '+ str(n))
		pdf.savefig()
		plt.close(f)
		'''
		for i in range(len(times_to_plot)):
			y = temp[i]
			# Add some random "jitter" to the x-axis
			x = np.random.normal(i, 0.04, size=len(y))
			ax.plot(x, y, 'r.', alpha=0.2)
		#ax.set_yscale('symlog')
		'''
	temp2 = {}
	f, ax = plt.subplots()
	for nn, n in enumerate(ncl):
		temp = []
		for nn in ncl[n]:
			temp.append(ncl[n][nn][-1])
		temp2[n] = temp
		ax.hist(temp, bins=100, color=colors[n], density=True, cumulative=True, histtype='step')
		ax.set_title('starting population: '+ str(n))
		ax.set_yscale('log')
		ax.set_xscale('log')
	pdf.savefig()
	plt.close(f)
	for t1 in temp2:
		for t2 in temp2:
			print(t1, t2, sts.ks_2samp(temp2[t1], temp2[t2]))

def divide_by_time(ncl):
	max_time = 366
	out = []
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

def get_correlation(clns, pdf, p1, p2):
	values1 = []
	values2 = []
	f, ax = plt.subplots()
	for c in clns:
		if p1 == 'population_size':
			values1.append(clns[c]['population_size'][-1])
			ax.set_xscale('symlog')
		if p2 == 'population_size':
			values2.append(clns[c]['population_size'][-1])
			ax.set_yscale('symlog')
		if p1 == 'doubling_rate':
			values1.append(clns[c]['doubling_rate'][0])
		if p2 == 'doubling_rate':
			values2.append(clns[c]['doubling_rate'][0])
		if p1 == 'metastasis_rate':
			values1.append(clns[c]['metastasis_rate'][0])
		if p2 == 'metastasis_rate':
			values2.append(clns[c]['metastasis_rate'][0])
		if p1 == 'same_features':
			values1.append(clns[c]['same_features'][0])
		if p2 == 'same_features':
			values2.append(clns[c]['same_features'][0])
		if p1 == 'treatment_effectiveness':
			values1.append(clns[c]['treatment'][0])
		if p2 == 'treatment_effectiveness':
			values2.append(clns[c]['treatment'][0])
		if p1 == 'death_rate':
			values1.append(clns[c]['death_rate'][0])
		if p2 == 'death_rate':
			values2.append(clns[c]['death_rate'][0])
		if p1 == 'total_clones':
			values1.append(clns[c]['n_clones'][0])
		if p2 == 'total_clones':
			values2.append(clns[c]['n_clones'][0])
	value1_n = [x/max(values1) for x in values1]
	value2_n = [x/max(values2) for x in values2]
	ax.plot(value1_n, value2_n, marker='.', linestyle='none')
	ax.set_xlabel(p1)
	ax.set_ylabel(p2)
	corrcoef = np.corrcoef(value1_n,value2_n)
	ax.set_title(corrcoef[1,1])
	pdf.savefig()
	plt.close(f)
	return corrcoef

def get_parent(dt,t, how_many):
	starting_pop = dt[0][0]
	previous_pop = []
	current_pop = []
	delta= []
	possible_parents = []
	for p in dt[t-1]:
		previous_pop.append(dt[t-1][p])
		current_pop.append(dt[t][p])
		delta.append(np.absolute(current_pop[-1]-previous_pop[-1]))
		if delta[-1] == 0 or delta[-1] ==starting_pop:
			possible_parents.append(p)

	if len(possible_parents) != how_many:
		sorted_delta = sorted(delta)
		possible_parents = []
		for h in range(how_many):
			idx = delta.index(sorted_delta[h])
			possible_parents.append(idx)
	children = []
	for pp in dt[t]:
		if pp not in dt[t-1]:
			children.append(pp)
	return possible_parents, children



def get_genetic_tree(dt):
	out = {}
	#for sp in dt:
	#	out[sp] = {}
	for r in dt:
		out[r] = {}
		for t in dt[r]['clones']:
			if t == 0: #starting condition
				first_clone = [dt[r]['clones'][t][0]]
				out[r][0] ={'pop_size': first_clone, 'children':{}}

			else:
				how_many_clones_now = len(dt[r]['clones'][t])
				how_many_clones_before = len(out[r])
				if how_many_clones_now == how_many_clones_before:
					for nc in dt[r]['clones'][t]:
						out[r][nc]['pop_size'].append(dt[r]['clones'][t][nc])
				else:
					how_many_new_clones =how_many_clones_now-how_many_clones_before
					parents, children =get_parent(dt[r]['clones'],t, how_many_new_clones)
					for c in dt[r]['clones'][t]:
						if dt[r]['clones'][t][c] == dt[r]['clones'][0][0]: # new clone
							out[r][c] = {}
							out[r][c]['pop_size'] = [dt[r]['clones'][t][c]]
							out[r][c]['children'] = {}
						else:
							out[r][c]['pop_size'].append(dt[r]['clones'][t][c])
							if c in parents:
								out[r][c]['children'][t] = children[parents.index(c)]
	return out


def in_newick_format(gt):
	out = {}
	for r in gt:
		out[r] = '('
		chwch = []
		root = gt[r][0]
		if len(root['children']) == 0: #single clone
			out[r] = 'single_clone'
		else:
			for t in root['children']:
				child_id = root['children'][t]
				child_letter = get_child_id(child_id)
				if len(gt[r][child_id]['children']) == 0: # it has no children
					out[r]+=str(child_letter)+','
				else:
					chwch.append(child_id)
					out[r]+='(...)'+ str(child_letter)+','

			while len(chwch)>0:
				newick_child, newcwc = get_newick(gt[r],chwch[0])
				temp = out[r].split('(...)'+str(get_child_id(chwch[0]))+',')
				if len(temp)!=2:
					print(temp)
					raise ValueError('something wrong with the split')
				out[r] = temp[0]+newick_child+temp[1]
				chwch.remove(chwch[0])
				for n in newcwc:
					chwch.append(n)
	for o in out:
		if 'single_clone' not in out[o]:
			out[o] = out[o][:-1]+');'
	return out


def get_child_id(n):
	if n < 1:
		raise ValueError("n must be a positive integer")
	result = ""

	while n > 0:
		n -= 1
		result = chr(ord('A') + n % 26) + result
		n //= 26

	return result


def get_newick(cgt, cid):
	out = '('
	new_chwch =[]
	for t in cgt[cid]['children']:
		gc = cgt[cid]['children'][t]
		if len(cgt[gc]['children'])==0:
			gc_letter = get_child_id(gc)
			out+=str(gc_letter)+','
		else:
			new_chwch.append(gc)
			gc_letter = get_child_id(gc)
			out += '(...)' + str(gc_letter)+','
	out+=str(get_child_id(cid))+')'+str(get_child_id(cid))+','
	return out,new_chwch

def get_clones_by_n_children(gt):
	out = {}
	temp = []
	for g in gt:
		temp.append(len(gt[g]))
	temp = sorted(list(set(temp)), reverse=True)
	for t in temp:
		idx = len(out)
		out[idx] = {'n_children': t, 'which_clones':[]}
		for g in gt:
			if len(gt[g])==t:
				out[idx]['which_clones'].append(g)
	return out


def print_genetic_trees(gtn, sd, fout):
	for g in gtn:
		filename = fout+'newick_genetic_tree_starting_concentration_'+str(sd)+'_replicate_'+str(g)+'.dnd'
		with open(filename, 'w') as F:
			F.write(gtn[g])


def print_clones_by_n_children(dt, fout):
	f, ax = plt.subplots()
	for d in dt:
		for c in dt[d]:
			for h in range(len(dt[d][c]['which_clones'])):
				x = d +0.3*d*np.random.random()
				ax.scatter(x,dt[d][c]['n_children'], color='b', marker='.')
	ax.set_xscale('symlog')
	ax.set_yscale('symlog')
	fout.savefig()

def get_n_children_by_parameters(gt, dt):
	out_doubling_rate={}
	out_met_rate={}
	out_same_feat={}
	out_treat={}
	out_death_rate={}
	temp = []
	for g in gt:
		temp.append(len(gt[g]))
	temp = sorted(list(set(temp)), reverse=True)
	for t in temp:
		idx = len(out_doubling_rate)
		out_doubling_rate[idx] = {'n_children': t, 'par_value': []}
		out_met_rate[idx] = {'n_children': t, 'par_value': []}
		out_same_feat[idx] = {'n_children': t, 'par_value': []}
		out_treat[idx] = {'n_children': t, 'par_value': []}
		out_death_rate[idx] = {'n_children': t, 'par_value': []}
		for g in gt:
			if len(gt[g]) == t:
				for c in range(t):
					out_doubling_rate[idx]['par_value'].append(dt[g]['doubling_rates'][365][c])
					out_met_rate[idx]['par_value'].append(dt[g]['metastasis_rates'][365][c])
					out_same_feat[idx]['par_value'].append(dt[g]['same_features'][365][c])
					out_treat[idx]['par_value'].append(dt[g]['treatment'][365][c])
					out_death_rate[idx]['par_value'].append(dt[g]['death_rate'][365][c])

	return out_doubling_rate,out_met_rate, out_same_feat, out_treat, out_death_rate

def print_children_by_parameter(dt,title, fout):
	for d in dt:
		f, ax = plt.subplots()
		to_plot = []
		for c in dt[d]:
			to_plot.append(dt[d][c]['par_value'])
		ax.boxplot(list(reversed(to_plot)))
		ax.set_title(title+' '+ str(d))
		fout.savefig()
		plt.close(f)


def get_clones_class(gt):
	out = {}
	for g in gt:
		if len(gt[g][0]['children'])==0: #single clone
			out[g] = 0
		else:
			cls = 1 #single generation
			for t in gt[g][0]['children']:
				clone_id = gt[g][0]['children'][t]
				if len(gt[g][clone_id]['children'])>0:
					cls = 2 # multiple generations
					break
			out[g] = cls
	return out

def get_total_pop_by_clones(gt):
	out = {}
	for g in gt:
		nc = len(gt[g])
		if nc not in out:
			out[nc] = []
		total_pop = 0
		start_pop = gt[g][0]['pop_size'][0]
		for c in gt[g]:
			total_pop+= gt[g][c]['pop_size'][-1]
		out[nc].append(total_pop/start_pop)
	return out

def plot_population_by_clones(totp, fout):
	for sp in totp:
		f, ax = plt.subplots()
		x = []
		m =[]
		s = []
		for c in totp[sp]:
			x.append(c)
			m.append(np.mean(totp[sp][c]))
			s.append(np.std(totp[sp][c]))
		ax.errorbar(x, m, yerr=s, linestyle='None', marker='d')
		ax.set_title('population by clones '+ str(sp))
		ax.set_xscale('symlog')
		ax.set_yscale('symlog')
		fout.savefig()
		plt.close(f)

def plot_clones_classes(cls, fout):
	f, ax =plt.subplots()
	to_plot_0 = []
	x_values = []
	to_plot_1 = []
	to_plot_2 = []

	start_x = 0
	for sp in sorted(cls):
		print(sp)
		n0=0
		n1=0
		n2=0
		for c in cls[sp]:
			if cls[sp][c]==0:
				n0+=1
			elif cls[sp][c] ==1:
				n1+=1
			elif cls[sp][c] ==2:
				n2+=1
			else:
				print(c)
				raise ValueError('what is happening?')
		to_plot_0.append(n0/(n0+n1+n2))
		to_plot_1.append(n1/(n0+n1+n2))
		to_plot_2.append(n2/(n0+n1+n2))
		x_values.append(start_x)
		start_x+=1
	ax.bar(x_values, to_plot_0)
	ax.bar(x_values, to_plot_1, bottom=to_plot_0)
	ax.bar(x_values, to_plot_2, bottom=[to_plot_0[ii]+i for ii, i in enumerate(to_plot_1)])
	fout.savefig()
	plt.close(f)


def reduce_clones(cls, cls_clas):
	out = {}
	for sp in cls:
		out[sp] = {}
		for c in cls[sp]:
			if cls_clas[sp][c]>0:
				out[sp][c] = cls[sp][c]
				out[sp][c]['class'] = cls_clas[sp][c]
	return out


def get_n_children_by_parameters2(gt, dt, cls):
	out_doubling_rate={0:{}, 1:{}, 2:{}}
	out_met_rate={0:{}, 1:{}, 2:{}}
	out_same_feat={0:{}, 1:{}, 2:{}}
	out_treat={0:{}, 1:{}, 2:{}}
	out_death_rate={0:{}, 1:{}, 2:{}}
	temp = []
	for g in gt:
		temp.append(len(gt[g]))
	temp = sorted(list(set(temp)), reverse=True)
	idx = 0
	for t in temp:
		for g in gt:
			if len(gt[g]) == t:
				if idx not in out_doubling_rate[cls[g]]:
					out_doubling_rate[cls[g]][idx] = {'n_children': t, 'par_value': []}
					out_met_rate[cls[g]][idx] = {'n_children': t, 'par_value': []}
					out_same_feat[cls[g]][idx] = {'n_children': t, 'par_value': []}
					out_treat[cls[g]][idx] = {'n_children': t, 'par_value': []}
					out_death_rate[cls[g]][idx] = {'n_children': t, 'par_value': []}

				for c in range(t):
					out_doubling_rate[cls[g]][idx]['par_value'].append(dt[g]['doubling_rates'][365][c])
					out_met_rate[cls[g]][idx]['par_value'].append(dt[g]['metastasis_rates'][365][c])
					out_same_feat[cls[g]][idx]['par_value'].append(dt[g]['same_features'][365][c])
					out_treat[cls[g]][idx]['par_value'].append(dt[g]['treatment'][365][c])
					out_death_rate[cls[g]][idx]['par_value'].append(dt[g]['death_rate'][365][c])
		idx+=1

	return out_doubling_rate,out_met_rate, out_same_feat, out_treat, out_death_rate


def print_children_by_parameter2(dt,title, fout):
	for d in dt:
		f, ax = plt.subplots(3,1)
		to_plot_0 = []
		to_plot_1 = []
		to_plot_2 = []
		for cl in dt[d]:
			for c in dt[d][cl]:
				if cl ==0:
					to_plot_0.append(dt[d][cl][c]['par_value'])
				elif cl ==1:
					to_plot_1.append(dt[d][cl][c]['par_value'])
				elif cl==2:
					to_plot_2.append(dt[d][cl][c]['par_value'])
				else:
					print(cl)
					raise ValueError('wrong class')
		ax[0].boxplot(list(reversed(to_plot_0)))
		ax[1].boxplot(list(reversed(to_plot_1)))
		ax[2].boxplot(list(reversed(to_plot_2)))
		ax[0].set_title(title+' '+ str(d))
		fout.savefig()
		plt.close(f)



folder_data = '/Users/marilisacortesi/Desktop/clonal_evolution/outputs/'
file_out = PdfPages('/Users/marilisacortesi/Desktop/clonal_evolution/plots.pdf')
list_files = os.listdir(folder_data)

data = {}
for f in list_files:
	if f.startswith('.'):
		continue
	starting_population = int(f.split('_')[2].split('_')[0])
	clone_id = int(f.split('_')[-1].split('.')[0])
	if starting_population not in data:
		data[starting_population] = {}
	with open(folder_data+f, 'rb') as F:
		data[starting_population][clone_id] = pkl.load(F)

n_clones, len_sims, max_clones= compute_n_clones(data)
plot_n_clones(n_clones, file_out)


'''
temp = []
sp = []
time_ticks = sorted(list(max_clones.keys()))
for m in time_ticks:
	print(m)
	temp.append(max_clones[m])
f, ax = plt.subplots()
ax.boxplot(temp)
ax.set_yscale('log')
file_out.savefig()

clones = divide_by_clones(data)

parameters = ['population_size', 'doubling_rate', 'metastasis_rate', 'same_features', 'treatment_effectiveness',
			  'death_rate', 'total_clones']
corr_values = {}
for p1 in parameters:
	corr_values[p1] = {}
	for p2 in parameters:
		corr_values[p1][p2]= get_correlation(clones, file_out, p1, p2)
		print(p1, p2, corr_values[p1][p2])


'''
clones_by_n_children = {}
clones_by_doubling_rate = {}
clones_by_met_rate = {}
clones_by_same_feat ={}
clones_by_treat = {}
clones_by_death_rate = {}

clones_by_doubling_rate_cls = {}
clones_by_met_rate_cls = {}
clones_by_same_feat_cls ={}
clones_by_treat_cls = {}
clones_by_death_rate_cls = {}
total_population_by_clones = {}
clones_classification={}
for d in data:
	genetic_tree = get_genetic_tree(data[d])
	total_population_by_clones[d] = get_total_pop_by_clones(genetic_tree)
	clones_classification[d] = get_clones_class(genetic_tree)
	clones_by_n_children[d] = get_clones_by_n_children(genetic_tree)
	clones_by_doubling_rate[d], clones_by_met_rate[d], clones_by_same_feat[d], clones_by_treat[d], clones_by_death_rate[d] = get_n_children_by_parameters(genetic_tree, data[d])
	clones_by_doubling_rate_cls[d], clones_by_met_rate_cls[d], clones_by_same_feat_cls[d], clones_by_treat_cls[d], clones_by_death_rate_cls[d] = get_n_children_by_parameters2(genetic_tree, data[d], clones_classification[d])
	genetic_tree_newick = in_newick_format(genetic_tree)
	print_genetic_trees(genetic_tree_newick, d, '/Users/marilisacortesi/Desktop/clonal_evolution/newick/')




print_clones_by_n_children(clones_by_n_children, file_out)
print_children_by_parameter(clones_by_doubling_rate,'doubling_rate', file_out)
print_children_by_parameter(clones_by_met_rate,'metastasis_rate', file_out)
print_children_by_parameter(clones_by_same_feat,'same features', file_out)
print_children_by_parameter(clones_by_treat,'treatment', file_out)
print_children_by_parameter(clones_by_death_rate,'death_rate', file_out)

print_children_by_parameter2(clones_by_doubling_rate_cls,'doubling_rate', file_out)
print_children_by_parameter2(clones_by_met_rate_cls,'metastasis_rate', file_out)
print_children_by_parameter2(clones_by_same_feat_cls,'same features', file_out)
print_children_by_parameter2(clones_by_treat_cls,'treatment', file_out)
print_children_by_parameter2(clones_by_death_rate_cls,'death_rate', file_out)
plot_population_by_clones(total_population_by_clones, file_out)
plot_clones_classes(clones_classification, file_out)

file_out.close()