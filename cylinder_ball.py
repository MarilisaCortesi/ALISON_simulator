import dill
import matplotlib.pyplot as plt

import ALISON.simulator
import ALISON.utility
import numpy as np
import pyvista as pv
import fenics



def add_core(struct, rad, val):
	center = np.mean(struct.precomputed_mesh_parameters.mesh.points, axis=0)
	for e in range(struct.precomputed_mesh_parameters.mesh.n_points):
		dist = np.linalg.norm(struct.precomputed_mesh_parameters.mesh.points[e, :] - center)
		if dist <= rad:
			struct.precomputed_mesh_parameters.mesh['glucose'][e] = val
		else:
			struct.precomputed_mesh_parameters.mesh['glucose'][e] = 0.1#struct.precomputed_mesh_parameters.mesh['glucose'][e]
	return struct


def get_time_points(list_times):
	output_variable = []
	for lt in list_times:
		try:
			output_variable.append(int(lt))
		except ValueError:
			output_variable.append(-1)
	return sorted(output_variable)


def get_mesh_values(data_time):
	mesh = data_time['mesh']
	out = []
	print(sum(mesh['glucose']))
	for g in mesh['oxygen']:
		#if g > 0:
		out.append(g)
	return out

def get_sections(mesh,field, n_slices):
	zetas = mesh.points[:, -1]
	min_z = min(zetas)
	max_z = max(zetas)
	slice_thickness = (max_z - min_z)/n_slices
	values = {}
	for i in range(n_slices):
		z_level = slice_thickness * i
		delta_z = np.absolute(zetas - z_level)
		points = []
		for idd, d in enumerate(delta_z):
			if d < 0.1:
				points.append(idd)
		values[i] = mesh[field][points]
	return values

# run the simulation
file_name = 'cylinder_ball.txt'
radius = 3
value = 0.5
structure = ALISON.simulator.ALISON(file_name)
structure = add_core(structure, radius, value)
#structure.precomputed_mesh_parameters.m.data = np.round(structure.precomputed_mesh_parameters.m.data, 3)
#structure.precomputed_mesh_parameters.k.data = np.round(structure.precomputed_mesh_parameters.k.data, 3)
structure.simulate()


'''
#analyse the data
fileName = 'outputs/31072022_10:40:03_complete_simulation.pickle' # 29072022_14:55:33_complete_simulation.pickle'
with open(fileName, 'rb') as F:
	data = dill.load(F)
time = get_time_points(list(data.keys()))


#reduced_time = [time[0], time[1], time[-1]]
#data_to_plot = []
#for t in reduced_time:
#	if t == -1:
#		key = 'initial_condition'
#	else:
#		key = t
#	data_to_plot.append(get_mesh_values(data[key]))
data_over_time = {}
for t in time:
	if t == -1:
		t = 'initial_condition'

	sections = get_sections(data[t]['mesh'], 'oxygen',2)
	for s in sections:
		if s not in data_over_time:
			data_over_time[s] = {'M': [], 'S': []}
		data_over_time[s]['M'].append(np.mean(sections[s]))
		data_over_time[s]['S'].append(np.std(sections[s]))
f, ax = plt.subplots(5,1)
colors = {0: 'b', 1: 'g', 2: 'm', 3: 'r', 4: 'k'}
for d in data_over_time:
	ax[d].errorbar(range(len(data_over_time[d]['M'])), data_over_time[d]['M'], yerr=data_over_time[d]['S'], color=colors[d])
plt.show()
'''