from ALISON import mesh_configurator as mesh
import dill

base_name = 'tetgen_80K_elements'

precomputed_parameters = mesh.Mesh(base_name)
file_out = precomputed_parameters.folder + base_name +'.pickle'
with open(file_out, 'wb') as F:
	dill.dump(precomputed_parameters, F)
