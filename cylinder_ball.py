import dill
import matplotlib.pyplot as plt

import ALISON.simulator
import ALISON.utility
import numpy as np
import pyvista as pv


def add_core(struct, rad, val):
	center = np.mean(struct.precomputed_mesh_parameters.mesh.points, axis=0)
	for e in range(struct.precomputed_mesh_parameters.mesh.n_points):
		dist = np.linalg.norm(struct.precomputed_mesh_parameters.mesh.points[e, :] - center)
		if dist <= rad:
			struct.precomputed_mesh_parameters.mesh['glucose'][e] = val
		else:
			struct.precomputed_mesh_parameters.mesh['glucose'][e] = 1#struct.precomputed_mesh_parameters.mesh['glucose'][e]
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
	print(sum(mesh['oxygen']))
	for g in mesh['oxygen']:
		#if g > 0:
		out.append(g)
	return out
'''
# run the simulation
file_name = 'cylinder_ball.txt'
radius = 3
value = 0.2
structure = ALISON.simulator.ALISON(file_name)
structure = add_core(structure, radius, value)
#structure.precomputed_mesh_parameters.m.data = np.round(structure.precomputed_mesh_parameters.m.data, 3)
#structure.precomputed_mesh_parameters.k.data = np.round(structure.precomputed_mesh_parameters.k.data, 3)
structure.simulate()


'''
#analyse the data
fileName = 'outputs/02062022_16:36:11_complete_simulation.pickle'
with open(fileName, 'rb') as F:
	data = dill.load(F)
time = get_time_points(list(data.keys()))
reduced_time = [time[0], time[1], time[-1]]
data_to_plot = []
for t in reduced_time:
	if t == -1:
		key = 'initial_condition'
	else:
		key = t
	data_to_plot.append(get_mesh_values(data[key]))

f, ax = plt.subplots()
ax.boxplot(data_to_plot)
plt.show()