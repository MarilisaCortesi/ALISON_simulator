from ALISON import simulator as organoid
import dill
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

	#configuration_file = 'organoid_configuration.txt'
	#structure = organoid.ALISON(configuration_file)
	#fileOut = 'debugging_structure.pickle'
	#with open(fileOut, 'wb') as F:
	#	dill.dump(structure, F)
	file_name = 'debugging_structure.pickle'
	with open(file_name, 'rb') as F:
		structure = dill.load(F)
	structure.precomputed_mesh_parameters.mesh.clear_field_data()
	for i in structure.initial_conditions:
		structure.precomputed_mesh_parameters.mesh.field_data[i] = structure.initial_conditions[i] + \
							 np.zeros(structure.precomputed_mesh_parameters.mesh.n_points)
	structure.precomputed_mesh_parameters.m = structure.precomputed_mesh_parameters.m*1000 # unit changed from g/mm3 to mg/mm3
	structure.simulate()
