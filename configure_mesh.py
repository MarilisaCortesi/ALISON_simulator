import mesh_configurator as mesh
import dill

base_name = 'cylinder_sources_sinks_small'

precomputed_parameters = mesh.Mesh(base_name)
file_out = precomputed_parameters.folder + base_name +'.pickle'
with open(file_out, 'wb') as F:
	dill.dump(precomputed_parameters, F)
