import os
import numpy as np
import scipy.sparse.linalg
from scipy.sparse import lil_array
import sympy
import fenics
import mshr
from ALISON import utility


def find_cell(cell_population, location):
	for c in cell_population:
		if c.location == location:
			return c


def get_material_properties(file_path):
	out = {}
	with open(file_path) as F:
		for r in F.readlines():
			if '#' not in r:
				temp = r.split('\n')[0].split('->')
				variable = temp[0].split('[')[0].strip()
				unit = temp[0].split('[')[1].split(']')[0].strip()
				has_time, unit = utility.contains_time_unit(unit)
				if has_time:
					value, unit2 = utility.convert_in_hours(temp[1] + unit)
				else:
					value = float(temp[1])
				out[variable] = value
	return out


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
	nodes = mesh.coordinates()
	if region == 'top':
		max_z = max(nodes[:, -1])
		return list(np.where(nodes[:, -1] == max_z)[0])
	elif region == 'bottom':
		min_z = min(nodes[:, -1])
		return list(np.where(nodes[:, -1] == min_z)[0])
	elif region == 'side':
		max_x = max(nodes[:, 0])
		tmp = nodes[:, 0:2] - np.array((0, 0))
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
				level_gml = float(r.split('->')[1].split('[')[0]) / 1000  # g/ml
				break
	gculture = 200 * level_gml  # each culture has a volume of 200 ul #TODO remove this magic number
	return gculture / mesh_elements


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
		return total_amount / mesh_elements
	else:
		new_amount = (float(o2_level) * total_amount) / 20  # the standard O2 level in an incubator is 20%
		return new_amount / mesh_elements


def initialize_f(mesh, fields, cells_population,vector_space):
	output_f = {}
	for ic in fields:
		temp = np.zeros((len(mesh.coordinates()[:, 0]))) # from the example in fenics it looks like f has the same length as the number of nodes
		for c in cells_population:
			temp[c.location] = get_rate(c, ic)
		output_f[ic] = fenics.Function(vector_space)
		output_f[ic].vector().set_local(temp)
	return output_f

def update_f(fields, cells_population,vector_space):
	output_f = {}
	test_field = fields['glucose'].vector().get_local()
	for ic in fields:
		temp = np.zeros_like(test_field) # from the example in fenics it looks like f has the same length as the number of nodes
		for c in cells_population:
			temp[c.location] = get_rate(c, ic)
		output_f[ic] = fenics.Function(vector_space)
		output_f[ic].vector().set_local(temp)
	return output_f
def initialise_mesh(resolution):
	cylinder = mshr.Cylinder(fenics.Point(0, 0, 0), fenics.Point(0, 0, 15), 5, 5) # these are the standard dimensions of a 96 well plate
	geometry = cylinder
	mesh = mshr.generate_mesh(geometry, resolution)  # 81.1 gives a mesh with ~ 80K nodes, while 44.7 yields one with ~80K elements
	return mesh

def get_f(precomputed_pars, el, ktbar, rate):  # TODO surface integrals have different integration points
	mesh = precomputed_pars.mesh
	out_f = np.zeros(mesh.n_points)
	# det_j = np.linalg.det(precomputed_pars.jacobian[el])
	integration_points = 4  # [[0.1381966, 0.1381966, 0.1381966], [0.58541020, 0.1381966, 0.1381966],
	# [0.1381966, 0.58541020, 0.1381966], [0.1381966, 0.1381966, 0.58541020]]
	w = 1 / 24  # 0.25#*precomputed_pars.volume_element #0416667
	ncp = (-1) * rate
	for jj, j in enumerate(precomputed_pars.nodes_in_elements[el]):
		temp1 = []
		for ip in range(integration_points):
			nj = precomputed_pars.ns[el][jj][ip]
			temp1.append(nj * w * ncp)
		out_f[j] = sum(temp1) + ktbar[el][jj]
	return out_f

def is_occupied(cell_pop, element):
	out1 = 0
	out2 = -1
	for ic, c in enumerate(cell_pop):
		if c.location == element:
			out1 = 1
			out2 = ic
			break
	return out1, out2


def get_fe(el, field, cell_population):
	io, idx = is_occupied(cell_population, el)
	if io:
		cell = cell_population[idx]
		qv = get_rate(cell, field)
	else:
		qv = 0
	return qv




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
					rate = (-1) * rate
				return rate


def compute_ktbar(k, boundary, idx, variable_level):
	output_variable = []
	kk = k[[idx], :].toarray()[0]
	for s in boundary['fixed_value']:
		output_variable.append(kk[s] * variable_level)
	return sum(output_variable)


def initialize_fields(initial_conditions, function_space):
	out = {}
	for i in initial_conditions:
		temp = fenics.Expression((str(initial_conditions[i])), degree=1) #Constant(initial_conditions[i] + np.zeros(len(mesh.coordinates()[:, 0])))
		#out[i] = fenics.Function(function_space)
		out[i] = fenics.interpolate(temp, function_space)
	return out


def get_field_predictor(parameters, f, boundary_conditions, fixed_flux):
	field_predictor = {}
	fields = parameters.mesh.array_names
	fields.remove('cells')
	m_precond = scipy.sparse.linalg.spilu(parameters.m_symbol)
	m_precond = scipy.sparse.linalg.LinearOperator(parameters.m_symbol.shape, m_precond.solve)
	for field in fields:
		temp = parameters.k_symbol.dot(parameters.mesh[field])
		temp2 = f[field] - temp
		[field_derivative, info] = scipy.sparse.linalg.gcrotmk(parameters.m_symbol, temp2,
															   x0=parameters.mesh[field])  # , show=True)
		if info != 0:
			raise ValueError('inversion_error field predictor. Info: ' + str(info))
		# field_derivative[field_derivative < 10e-3] = 0
		# field_derivative = parameters.inv_m.dot(temp2)
		# field_derivative[boundary_conditions['fixed_flux']] = fixed_flux
		field_predictor[field] = parameters.mesh[field] + 0.5 * field_derivative  # 0.01 is deltat
	return field_predictor


def update_environment(cell, fields, vector_space):
	print('s')
	out = {}
	for f in fields:
		rate = get_rate(cell, f)
		temp_field = fields[f].vector().get_local()
		temp_field[cell.location] += rate
		out[f] = fenics.Function(vector_space)
		out[f].vector().set_local(temp_field)
	return out



def boundary(x, on_boundary):
	tol = 1E-14
	return on_boundary and fenics.near(x[0-1], 15, tol)


def set_boundary_conditions(initial_conds, function_space):
	out = {}
	for i in initial_conds:
		out[i] = fenics.DirichletBC(function_space, initial_conds[i], boundary)
	return out

def recompute_f(precomputed_pars, field, boundary_conditions, cell_population, init_cond):
	output_f = np.zeros((precomputed_pars.mesh.n_points))
	for e in range(precomputed_pars.mesh.n_faces):
		fe = get_fe(precomputed_pars, boundary_conditions, e, field, cell_population, init_cond)
		nodes_element = precomputed_pars.nodes_in_elements[e]
		for ii, i in enumerate(nodes_element):
			output_f[i] += fe[ii][0]
	return output_f

def get_production_consumption(cell_pop, e, field):
	rate = 0
	for c in cell_pop:
		if c.location == e:
			rate = get_rate(c, field)
			break
	return rate


def solve_diffusion(precomputed_pars, field_predictor, boundary_condition, fixed_flux, init_cond, field,
					cell_population):
	f = recompute_f(precomputed_pars, field, boundary_condition, cell_population, init_cond)
	el_1 = precomputed_pars.k_symbol.dot(field_predictor[field])
	a_matrix = scipy.sparse.csc_matrix(precomputed_pars.m_symbol + 0.5 * precomputed_pars.k_symbol)
	b_matrix = (0.5 * 1 * f + precomputed_pars.m_symbol.dot(precomputed_pars.mesh[field]))
	a_precond = scipy.sparse.linalg.spilu(a_matrix)
	a_precond = scipy.sparse.linalg.LinearOperator(a_matrix.shape, a_precond.solve)
	x0 = precomputed_pars.mesh[field]
	[field_values, info] = scipy.sparse.linalg.gcrotmk(a_matrix, b_matrix, x0=x0)
	if info != 0:
		raise ValueError('inversion error, info: ' + str(info))
	# for ff in range(len(field_values)):
	#	field_values[ff] = round(field_values[ff], 3)  #necessary to avoid drift in homogeneous fields
	# field_values[np.where(field_values < 0)] = 0
	test = field_values[boundary_condition['fixed_flux']]
	field_values[boundary_condition['fixed_value']] = init_cond
	# [field_derivative, info_der] = scipy.sparse.linalg.cgs(precomputed_pars.m, f-el_1)
	field_derivative = (field_values - field_predictor[field]) / (0.5)
	# field_derivative[field_derivative < 1e-3] = 0

	field_values[np.where(field_values < 0)] = 0
	'''
	field_derivative = inv_m05k.dot(f - el_1)
	#field_derivative[boundary_condition['fixed_flux']] = fixed_flux
	field = field_predictor[field] + 0.5 * 1 * field_derivative
	field[np.where(field < 0)] = 0
	field[boundary_condition['fixed_value']] = init_cond
	'''
	field_predictor = field_values + 0.5 * field_derivative
	# field_predictor[np.where(field_predictor<0 )] =0

	test2 = sum(field_values)
	return field_values, field_predictor


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
