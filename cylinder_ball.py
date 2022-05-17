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
	return mesh['glucose']
'''
# run the simulation
file_name = 'cylinder_ball.txt'
radius = 2
value = 5
structure = ALISON.simulator.ALISON(file_name)
structure = add_core(structure, radius, value)
file_out = 'cylinder_ball_init.pickle'
with open(file_out, 'wb') as F:
	dill.dump(structure, F)
'''
fileName = 'cylinder_ball_init.pickle'
with open(fileName, 'rb') as F:
	structure = dill.load(F)
structure.simulate()

'''
#analyse the data
fileName = 'outputs/17052022_15:42:38_complete_simulation.pickle'
with open(fileName, 'rb') as F:
	data = dill.load(F)
time = get_time_points(list(data.keys()))
data_to_plot = []
for t in time:
	if t == -1:
		key = 'initial_condition'
	else:
		key = t
	data_to_plot.append(get_mesh_values(data[key]))
f, ax = plt.subplots()
ax.boxplot(data_to_plot)
plt.show()
'''





