import os
import numpy as np
import scipy.sparse.linalg
from ALISON import utility


def find_cell(cell_population, location):
	for c in cell_population:
		if c.location == location:
			return c


def get_material_properties(mat_prop):
	output_variable = {'rho': None, 'cv': None, 'k': None}
	for m in mat_prop:
		if m in output_variable:
			output_variable[m] = float(mat_prop[m])
	return output_variable


def get_boundary_conditions(mesh_conf, mesh):
	output_variable = {'fixed_value': [], 'fixed_flux': []}
	if 'fixed value' in mesh_conf:
		if type(mesh_conf['fixed value']) == str:
			output_variable['fixed_value'] = get_elements_mesh(mesh, mesh_conf['fixed value'])
		elif type(mesh_conf['fixed value']) == list:
			for f in mesh_conf['fixed value']:
				output_variable['fixed_value'] += get_elements_mesh(mesh, f)
		else:
			raise ValueError('unrecognized fixed value type')
	if 'fixed flux' in mesh_conf:
		if type(mesh_conf['fixed flux']) == str:
			output_variable['fixed_flux'] = get_elements_mesh(mesh, mesh_conf['fixed flux'])
		elif type(mesh_conf['fixed flux']) == list:
			for f in mesh_conf['fixed flux']:
				output_variable['fixed_flux'] += get_elements_mesh(mesh, f)
		else:
			raise ValueError('unrecognized fixed value type')
	return output_variable


def get_drug_level(configuration, mesh_elements):
	print('todo')


def get_elements_mesh(mesh, region):
	if region == 'top':
		max_z = max(mesh.points[:, 2])
		return list(np.where(mesh.points[:, 2] == max_z)[0])
	elif region == 'bottom':
		min_z = min(mesh.points[:, 2])
		return list(np.where(mesh.points[:, 2] == min_z)[0])
	elif region == 'side':
		max_x = max(mesh.points[:, 0])
		tmp = mesh.points[:, 0:2] - np.array((0, 0))
		side_nodes = []
		for tt, t in enumerate(tmp):
			distance = np.linalg.norm(t)
			if np.absolute(distance - max_x) < 1e-5:
				side_nodes.append(tt)
		return list(side_nodes)


def get_glucose_level(media, mesh_elements):
	full_path = os.getcwd() + os.path.sep + 'media' + os.path.sep + media + '.txt'
	with open(full_path) as F:
		for r in F.readlines():
			if 'glucose' in r:
				level_gml = 1000 * float(r.split('->')[1].split('[')[0])  # g/ml in ug/ul
				break
	gculture = 200 * level_gml  #each culture has a volume of 200 ul
	return gculture/mesh_elements


def get_initial_conditions(configuration, mesh_elements):
	# it hypothesizes the presence of at most 4 fields types (oxygen, glucose, lactate and drug) The drug one can contain
	# multiple treatments and is not mandatory
	output_initial_conditions = {'oxygen': None, 'glucose': None, 'lactate': 0}
	output_fixed_flux = 0
	output_initial_conditions['oxygen'] = get_oxygen_level(configuration['oxygen'], mesh_elements)
	output_initial_conditions['glucose'] = get_glucose_level(configuration['media'], mesh_elements)
	if configuration['treatment'] != 'none':
		output_initial_conditions['drug'] = get_drug_level(configuration['drug'], mesh_elements)
	return output_initial_conditions, output_fixed_flux


def get_oxygen_level(o2_level, mesh_elements):
	total_amount = 248.2  # see notebook 3 for the calculation
	if o2_level == 'standard':
		return total_amount/mesh_elements
	else:
		new_amount = (float(o2_level) * total_amount)/20  # the standard O2 level in an incubator is 20%
		return new_amount/mesh_elements


def initialize_f_ktbar(precomputed_parameters, initial_conditions, boundary_conditions, cells_population):
	output_f = {}
	output_ktbar = {}
	for ic in initial_conditions:
		output_f[ic] = np.zeros(precomputed_parameters.mesh.n_points)
		output_ktbar[ic] = {}
		for e in range(precomputed_parameters.mesh.n_faces):
			fe, output_ktbar[ic][e] = get_fe(precomputed_parameters, boundary_conditions, e, cells_population,
											 initial_conditions, ic)
			output_f[ic] += fe
	return output_f, output_ktbar

def get_f(precomputed_pars, el, ktbar, rate):
	mesh = precomputed_pars.mesh
	out_f = np.zeros(mesh.n_points)
	det_j = np.linalg.det(precomputed_pars.jacobian[el])
	integration_points = 4  # [[0.1381966, 0.1381966, 0.1381966], [0.58541020, 0.1381966, 0.1381966],
	# [0.1381966, 0.58541020, 0.1381966], [0.1381966, 0.1381966, 0.58541020]]
	w = 0.0416667
	ncp = (-1) * rate
	for jj, j in enumerate(precomputed_pars.nodes_in_elements[el]):
		temp1 = []
		for ip in range(integration_points):
			nj = precomputed_pars.ns[el][jj][ip]
			temp1.append(nj * det_j * w * ncp)
		out_f[j] = sum(temp1) + ktbar[el][jj]
	return out_f


def get_fe(precomputed_pars, boundary, el, cell_pop, init_cnd, field):
	output_f = np.zeros(precomputed_pars.mesh.n_points)
	output_k = []
	det_j = np.linalg.det(precomputed_pars.jacobian[el])
	integration_points = 4 #TODO is there a way to remove this magic number?
	w = 0.0416667 # TODO does this depend on the number of integration points?
	if precomputed_pars.mesh['cells'][el] == 1:
		cell = find_cell(cell_pop, el)
		ncp = get_rate(cell, field)
	else:
		ncp = 0
	for jj, j in enumerate(precomputed_pars.nodes_in_elements[el]):
		dummy = []
		k_tb = compute_ktbar(precomputed_pars.k, boundary, j, init_cnd[field])
		output_k.append(k_tb)
		for ip in range(integration_points):
			nj = precomputed_pars.ns[el][jj][ip]
			dummy.append(nj * det_j * w * ncp)
		output_f[j] = sum(dummy) - k_tb
	return output_f, output_k


def get_rate(cell, field):
	if len(cell.rules['current_rules']['environment']) == 0:
		return 0
	else:
		for r in cell.rules['current_rules']['environment']:
			if cell.rules['current_rules']['environment'][r]['variable'] == field:
				par = cell.rules['current_rules']['environment'][r]['rate']
				try:
					rate = float(par)
				except ValueError:
					if par in cell.parameters:
						rate = cell.parameters[par]
					else:
						rate = utility.solve_equation(par, cell)
				if cell.rules['current_rules']['environment'][r]['action'] == 'consumption':
					rate = (-1)*rate
				return rate


def compute_ktbar(k, boundary, idx, variable_level):
	output_variable = []
	kk = k[[idx], :].toarray()[0]
	for s in boundary['fixed_value']:
		output_variable.append(kk[s] * variable_level)
	return sum(output_variable)


def initialize_fields(mesh, initial_conditions, boundary, fixed_value, fixed_flux):
	for i in initial_conditions:
		mesh.field_data[i] = initial_conditions[i] + np.zeros(mesh.n_points)
		mesh[i][boundary['fixed_value']] = fixed_value[i]
		mesh[i][boundary['fixed_flux']] = fixed_flux
	return mesh


def get_field_predictor(parameters, f):
	field_predictor = {}
	fields = parameters.mesh.array_names
	fields.remove('cells')
	for field in fields:
		temp = parameters.k.dot(parameters.mesh[field])
		field_derivative = scipy.sparse.linalg.inv(parameters.m).dot(f[field] - temp)
		field_predictor[field] = parameters.mesh[field] + 0.5 * field_derivative
	return field_predictor

def update_environment(precomputed_pars, field_predictor, boundary_conditions, fixed_flux, initial_conditions,
					   fd_1, ktbar, cell_population):
	mesh = precomputed_pars.mesh
	fields = mesh.array_names
	fields.remove('cells')
	field_predictor_out ={}
	for field in fields:
		temp_field, field_predictor_out[field] = solve_diffusion(precomputed_pars, field_predictor, boundary_conditions, fd_1, fixed_flux, initial_conditions[field], field,ktbar, cell_population)
		for xx, x in enumerate(temp_field):
			mesh[field][xx] = x
	return mesh, field_predictor_out


def recompute_f(precomputed_pars, field, ktbar, cell_population):
	mesh = precomputed_pars.mesh
	tmp = np.zeros(mesh.n_points)
	for e in range(mesh.n_faces):
		production_consumption = get_production_consumption(cell_population, e, field)
		fe = get_f(precomputed_pars, e, ktbar[field], production_consumption)
		tmp += fe
	return tmp


def get_production_consumption(cell_pop, e, field):
	rate = 0
	for c in cell_pop:
		if cell_pop.location == e:
			rate = get_rate(c, field)
			break
	return rate


def solve_diffusion(precomputed_pars, field_predictor, boundary_condition, fd_1, fixed_flux, init_cond, field, ktbar, cell_population):
	f = recompute_f(precomputed_pars, field, ktbar, cell_population)
	el_1 = precomputed_pars.k.dot(field_predictor[field])
	field_derivative = fd_1.dot(f - el_1)
	field_derivative[boundary_condition['fixed_flux']] = fixed_flux
	field = field_predictor[field] + 0.5 * field_derivative
	field[np.where(field < 0)] = 0
	field[boundary_condition['fixed_value']] = init_cond
	field_predictor = field + 0.5 * field_derivative
	return field, field_predictor

'''
class FEM:



	def simulate(self, iterations, fixed_values, fixed_flux, consumption_production):
		timestep = 1
		fields = self.get_fields(self.mesh)
		field_keeper = {}
		inv_m = scipy.sparse.linalg.inv(self.m)
		fd_1 = scipy.sparse.linalg.inv(self.m + 0.5 * timestep * self.k)
		for flds in fields:
			dot_1 = self.k.dot(self.mesh[flds])
			field_derivative = inv_m.dot(self.f[flds] - dot_1)
			field_predictor = self.mesh[flds] + 0.5 * timestep * field_derivative
			field_keeper[flds] = {}
			field_keeper[flds][0] = self.mesh[flds]
			precomputed_pars = {'nodes_el': self.nodes_in_elements, 'jacobian': self.jacobian,
								'material': self.material,
								'ns': self.ns, 'ktbar': self.k_tbar[flds]}
			for t in range(1,
						   iterations):  # TODO check that this is correct. I changed around a few things and I'm not sure anymore.
				self.f[flds] = self.recompute_f(self.mesh, precomputed_pars, consumption_production[flds])
				dot_2 = self.k.dot(field_predictor)
				field_derivative = fd_1.dot(dot_2)
				field_derivative[
					self.bc['fixed_flux']] = fixed_flux[flds]
				self.mesh[flds] = field_predictor + 0.5 * timestep * field_derivative
				if any(self.mesh[flds] < 0):
					self.mesh[flds][np.where(self.mesh[flds] < 0)] = 0
				self.mesh[flds][
					self.bc['fixed_value']] = fixed_values[flds]
				field_keeper[flds][t] = self.mesh[flds]
				field_predictor = self.mesh[flds] + 0.5 * timestep * field_derivative
		return field_keeper

	@staticmethod
	def get_fields(mesh):
		fields = mesh.array_names
		fields.remove('cells')
		return fields

	@staticmethod
	def recompute_f(mesh, precomputed_pars, production_consumption):
		tmp = np.zeros(mesh.n_points)
		for e in range(mesh.n_faces):
			fe = FEM.get_f(mesh, e, precomputed_pars, production_consumption)
			tmp += fe
		return tmp

	@staticmethod
	def get_f(mesh, el, precomputed_pars, production_consumption):
		out_f = np.zeros(mesh.n_points)
		det_j = np.linalg.det(precomputed_pars['jacobian'][el])
		integration_points = 4  # [[0.1381966, 0.1381966, 0.1381966], [0.58541020, 0.1381966, 0.1381966],
		# [0.1381966, 0.58541020, 0.1381966], [0.1381966, 0.1381966, 0.58541020]]
		w = 0.0416667
		nc = sum(mesh['cells'])
		p = (-1) * production_consumption
		ncp = nc * p * mesh['cells'][el]
		for jj, j in enumerate(precomputed_pars['nodes_el'][el]):
			temp1 = []
			for ip in range(integration_points):
				nj = precomputed_pars['ns'][el][jj][ip]
				temp1.append(nj * det_j * w * ncp)
			out_f[j] = sum(temp1) + precomputed_pars['ktbar'][el][jj]
		return out_f
'''