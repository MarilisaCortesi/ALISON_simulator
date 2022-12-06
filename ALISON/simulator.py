import itertools
import math
import os
import shutil
import datetime
import pickle
import dill
import fenics
import ufl
import tqdm
import numpy as np
import numpy.random as random
from scipy.sparse import lil_array
import scipy.sparse.linalg
from ALISON import utility
from ALISON import DS3FE
from ALISON import cells


class ALISON:
	def __init__(self, configuration_file):
		self.base_name = configuration_file.split('.txt')[0]
		full_path_configuration = os.getcwd() + os.path.sep + 'experiment_configuration_files' + os.path.sep + self.base_name + '.txt'
		self.experiment_configuration, self.structure_configuration, self.cells_configuration = \
			self.read_main_configuration_file(full_path_configuration)
		self.mesh = DS3FE.initialise_mesh(float(self.structure_configuration['resolution']))
		print('computing neighbouring nodes')
		self.neighbours = self.get_neighbours()
		self.function_space = fenics.FunctionSpace(self.mesh, 'P', 1)

		self.cell_population = self.add_cells(self.cells_configuration, self.structure_configuration, self.mesh)
		self.initial_conditions, self.fixed_flux = DS3FE.get_initial_conditions(self.experiment_configuration,
																				self.mesh.num_cells())
		self.boundary_conditions = DS3FE.set_boundary_conditions(self.initial_conditions, self.function_space)
		self.f = DS3FE.initialize_f(self.mesh, list(self.initial_conditions.keys()), self.cell_population,
									self.function_space)
		self.fields = DS3FE.initialize_fields(self.initial_conditions, self.function_space)

	def get_neighbours(self, threshold=0.5):
		coordinates = self.mesh.coordinates()
		out = {}
		for c in tqdm.tqdm(range(len(coordinates[:, 0]))):
			difference = coordinates - coordinates[c, :]
			norm = np.linalg.norm(difference, axis=1)
			out[c] = np.where(norm < threshold)[0]
		return out

	def simulate(self, out_name):
		# function that runs the simulation. #TODO: modify when integrating the FEM.
		iterations, resolution, unit = ALISON.get_iterations(self.experiment_configuration)
		simulation_folder = self.initialize_outputs(self.cell_population, self.fields,
													self.base_name)
		trial_function = fenics.TrialFunction(self.function_space)
		test_function = fenics.TestFunction(self.function_space)
		diff_coeff = fenics.Constant(
			float(self.structure_configuration['k']) / (float(self.structure_configuration['cv'])
														* float(self.structure_configuration['rho'])))
		dx = fenics.Measure('dx')
		F = {}
		a = {}
		L = {}
		for f in self.fields:
			F[f] = trial_function * test_function * dx + diff_coeff * resolution * fenics.dot(
				fenics.grad(trial_function),
				fenics.grad(test_function)) * dx - (
						   self.fields[f] + resolution * diff_coeff * self.f[f]) * test_function * dx
			a[f], L[f] = fenics.lhs(F[f]), fenics.rhs(F[f])
		trial_function = fenics.Function(self.function_space)
		t = 0
		for n in range(iterations):
			t += resolution
			trial_function_e = {}
			for v in F:
				fenics.solve(a[v] == L[v], trial_function, self.boundary_conditions[v])
				trial_function_e[v] = fenics.interpolate(trial_function, self.function_space)
				self.fields[v].assign(trial_function)
				#self.fields[v] = fenics.interpolate(self.fields[v], self.function_space)

			update_order = self.get_order(self.cell_population)  # order with which the cells are updated
			new_cell_population = self.cell_population.copy()
			for oo in tqdm.tqdm(range(len(update_order))):
				o = update_order[oo]
				if self.cell_population[o].type =='fibroblasts':
					check_neighbourhood = ALISON.get_local(self.cell_population, 'cancer', self.cell_population[o], self.neighbours)
					if check_neighbourhood > 0:
						self.cell_population[o].time_since_cancer_in_neighbourhood += 1
					else:
						if self.cell_population[o].time_since_cancer_in_neighbourhood > 0:
							self.cell_population[o].time_since_cancer_in_neighbourhood -= 0.5 #TODO: does this make sense?

				probability_vector = self.get_probabilities(self.cell_population[o], t,
															self.mesh, self.fields, self.neighbours,
															self.cell_population)
				to_execute = self.choose_rule(probability_vector)
				log = self.execute_rule(self.cell_population[o], to_execute, self.mesh,
										self.initial_conditions, self.neighbours, t, self.cell_population, self.fields)
				if log['is_new_cell']:
					new_cell_population.append(log['new_cell'])
				if log['executed_rule'] == 'degradation':
					new_cell_population.pop(o)
			self.cell_population = new_cell_population
			self.update_tracking_variables(simulation_folder, self.base_name, t,
										   self.fields,
										   self.cell_population)

			self.f = DS3FE.update_f(self.fields, self.cell_population, self.function_space)
			F = {}
			a = {}
			L = {}
			trial_function = fenics.TrialFunction(self.function_space)
			for f in self.fields:
				F[f] = trial_function * test_function * dx + diff_coeff * resolution * fenics.dot(
					fenics.grad(trial_function),
					fenics.grad(test_function)) * dx - (
							   self.fields[f] + resolution * diff_coeff * self.f[f]) * test_function * dx
				a[f], L[f] = fenics.lhs(F[f]), fenics.rhs(F[f])
			trial_function = fenics.Function(self.function_space)
			trial_function_e = fenics.interpolate(trial_function, self.function_space)
			self.fields[f].assign(trial_function_e)
		self.save_output(simulation_folder, out_name, unit)


	@staticmethod
	def update_cells_position(mesh, cls):
		for ic, c in enumerate(mesh['cells']):
			is_cell = ALISON.is_occupied(cls, ic)
			if is_cell == 1:
				if c == 0:
					mesh['cells'][ic] = 1
			else:
				if c == 1:
					mesh['cells'][ic] = 0
		return mesh

	@staticmethod
	def is_occupied(cells, location):
		out_variable = 0
		for c in cells:
			if c.location == location:
				out_variable = 1
				break
		return out_variable

	@staticmethod
	def save_output(output_folder, out_name, unit):
		files = os.listdir(output_folder)
		complete_simulation = {}
		for f in files:
			time = utility.convert_in_original_unit(int(float(f.split('=')[1].split('.pickle')[0])), unit)
			if time < 0:
				time = 'initial_condition'
			with open(output_folder + os.path.sep + f, 'rb') as F:
				mesh, cell_population = pickle.load(F)
				cell_population = utility.update_measurement_unit(cell_population, unit)
				complete_simulation[time] = {'mesh': mesh, 'cell_population': cell_population}
		file_name = output_folder.split('_current')[0] + '_' + out_name + '_complete_simulation.pickle'
		with open(file_name, 'wb') as F:
			pickle.dump(complete_simulation, F)
		shutil.rmtree(output_folder, ignore_errors=True)

	@staticmethod
	# function that looks for an empty neighbour of the current element.
	def find_empty_neighbour(mesh, initial_condition, neighbours, current_location, cell_population, fields):
		neighbouring_cells = neighbours[current_location]
		empty_neighbours = []
		scores = []
		for n in neighbouring_cells:
			if not ALISON.is_occupied(cell_population, n):
				scores.append(ALISON.get_score(fields, initial_condition, n, mesh.coordinates()))
				empty_neighbours.append(n)
		# TODO add neighbour score, which layer
		if len(empty_neighbours) > 0:
			max_score = max(scores)
			idxs_max = np.where(scores == max_score)[0] #[ii for ii, i in enumerate(scores) if i == max_score]
			if len(idxs_max) > 1:
				x = np.random.randint(0, len(idxs_max))
			else:
				x = idxs_max[0]
			return empty_neighbours[x]

	@staticmethod
	def get_score(fields, initial_condition, element, coords):
		glucose = fields['glucose'].vector().get_local()
		oxygen = fields['oxygen'].vector().get_local()
		lactate = fields['lactate'].vector().get_local()
		s_glucose = glucose[element] / initial_condition['glucose']
		s_oxygen = oxygen[element] / initial_condition['oxygen']
		s_lactate = lactate[element] / max(lactate)
		s_env = s_glucose + s_oxygen - s_lactate  # TODO check if it's ok
		z_el = coords[element][-1]
		min_z = min(coords[:, -1])
		max_z = max(coords[:, -1])
		s_pos = (max_z - z_el) / (max_z - min_z)
		return (s_pos + s_env) / 2

	@staticmethod
	# function that executes the chosen rule.
	def execute_rule(cell, te, mesh, initial_condition, neighbours, iteration, cell_population, fields):
		output_variable = {}
		try:
			chosen_rule = list(cell.rules['current_rules']['behaviour'])[te]
			result = cell.rules['current_rules']['behaviour'][chosen_rule]['end']
		except IndexError:  # the rule that maintains the cell in the current state is not in the configuration file.
			result = cell.status
		if type(result) == list:
			if result[0] == cell.status:  # doubling
				output_variable['executed_rule'] = 'doubling'
				output_variable['is_new_cell'] = 1
				new_location = ALISON.find_empty_neighbour(mesh, initial_condition, neighbours, cell.location,
														   cell_population, fields)
				cell.double()
				if cell.type == 'cancer':
					output_variable['new_cell'] = cells.CancerCell(new_location, cell.configuration, cell.status)
				elif cell.type == 'fibroblasts':
					output_variable['new_cell'] = cells.Fibroblast(new_location, cell.configuration, cell.status)
				elif cell.type == 'mesothelial':
					output_variable['new_cell'] = cells.MesothelialCell(new_location, cell.configuration, cell.status)
				else:
					raise ValueError('unrecognized cell type')
			elif result[0] == 0:  # migration
				output_variable['executed_rule'] = 'migration'
				output_variable['is_new_cell'] = 0
				new_location = ALISON.find_empty_neighbour(mesh, initial_condition, neighbours, cell.location,
														   cell_population, fields)
				cell.migrate(new_location)
			else:
				raise ValueError('unrecognized two voxels operation')
		else:
			if result == cell.status:  # stay in current state
				output_variable['executed_rule'] = 'stay'
				output_variable['is_new_cell'] = 0
				cell.stay()
			else:
				if result == 0:
					output_variable['executed_rule'] = 'degradation'
					output_variable['is_new_cell'] = 0
				else:
					output_variable['executed_rule'] = 'transition_to_other_state'
					output_variable['is_new_cell'] = 0
					cell.transition(result, iteration)
		return output_variable

	@staticmethod
	def update_tracking_variables(simulation_folder, base_name, time, fields, cell_population):
		now = datetime.datetime.now()
		file_name = now.strftime("%d_%m_%Y_%H:%M:%S") + '_current_simulation_' + base_name + '_T =' + str(
			time) + '.pickle'
		fields_out = {}
		for f in fields:
			fields_out[f] = fields[f].vector().get_local()
		with open(simulation_folder + file_name, 'wb') as f:
			pickle.dump([fields_out, cell_population], f)

	@staticmethod
	# function that chooses which rule to execute.
	def choose_rule(probability_vector):
		cumulative = np.cumsum(probability_vector)
		if cumulative[-1] > 1:
			raise ValueError('The sum of probabilities is above 1')
		rd_prob = np.random.random()
		tmp = cumulative - rd_prob < 0
		index = np.where(tmp == False)[0][0]
		return index

	@staticmethod
	def check_eligibility(neighbours, population):
		out = False
		for n in neighbours:
			if not ALISON.is_occupied(population, n):
				out = True
				break
		return out

	@staticmethod
	def get_probabilities(cll, t, mesh, fields, neighbours, cell_population):
		# function that gets the probability of a rule
		out = []
		for r in sorted(cll.rules['current_rules']['behaviour']):
			if type(cll.rules['current_rules']['behaviour'][r]['end']) == list:
				eligible = ALISON.check_eligibility(neighbours[cll.location], cell_population)
				if eligible:
					value = ALISON.get_value(cll, r, t, fields, cell_population, neighbours)
				else:
					value = 0
			else:
				value = ALISON.get_value(cll, r, t, fields, cell_population, neighbours)
			if value > 1 or value < 0:
				raise ValueError('probability above 1 or below 0 for rule ' + str(r))
			out.append(value)
		if sum(out) > 1:
			raise ValueError('sum of probability vector above 1')
		out.append(1 - sum(out))  # probability of remaining in the same state
		return out

	@staticmethod
	def get_value(cll, r, t, fields, cell_population, neighbours):
		# another function for the interpretation of the probability strings
		probability_string = cll.rules['current_rules']['behaviour'][r]['probability']
		parameter_values = cll.parameters
		operators_types = ['*', '/', '+'] #, '-']
		n_ops = 0
		for p in probability_string:
			if p in operators_types:
				n_ops += 1
		exec_op = 0
		while exec_op < n_ops:
			next_op, op1, op2, idxs = ALISON.find_next_operation(operators_types, probability_string)
			probability_string = ALISON.execute_operation(op1, op2, next_op, idxs, cll, t, probability_string,
														  parameter_values, fields, cell_population, neighbours)
			exec_op += 1
		if 'e' in probability_string:
			temp = probability_string.split('.')
			if len(temp) == 2:
				probability_string = temp[0]
			else:
				raise ValueError('check this out', probability_string)
		return float(probability_string)

	@staticmethod
	# function that gets the order of cell addition to the mesh
	def get_order(population):
		order = list(range(len(population)))
		random.shuffle(order)
		return order

	@staticmethod
	# function that determines which operation to do next
	def find_next_operation(optype, pstring):
		for o in optype:
			temp = pstring.split(o)
			if len(temp) > 1:
				op_out = o
				tmp2 = [len(x) for x in temp]
				op_pos = tmp2[0]
				op1, op2 = ALISON.get_operators(op_pos, optype, pstring)
				return op_out, op1, op2, [op_pos - len(op1), len(op2) + op_pos]

	@staticmethod
	# function that gets the operators for a defined operation #TODO: support for one operator operations.
	def get_operators(op, op_all, ps):
		o1_temp = ps[:op]
		o2_temp = ps[op + 1:]
		if ALISON.has_other_operators(o1_temp, op_all):
			ops = []
			for i in range(len(o1_temp)):
				if o1_temp[i] in op_all:
					ops.append(i)
			o1 = o1_temp[ops[-1] + 1:]
		else:
			o1 = o1_temp
		for i in range(len(o2_temp)):
			if o2_temp[i] in op_all:
				o2 = o2_temp[: i]
				break
			else:
				o2 = o2_temp
		return o1, o2

	@staticmethod
	# function that determines if a probability string has other operators.
	def has_other_operators(str, ops):
		out = 0
		for s in str:
			if s in ops:
				out = 1
				break
		return out

	@staticmethod
	# function that gets the value for an operator.
	def get_operator_value(opr, pars, cll, t, fields, cell_population, neighbours):
		if opr in pars:
			return pars[opr]
		else:
			try:
				return float(opr)
			except ValueError:
				if 'time' in opr:  # time, time_death, time_since_last_division
					if 'death' in opr:
						if cll.time_death is None:
							raise ValueError("This cell is not dead")
						else:
							return cll.time_death
					elif 'division' in opr:
						return cll.time_since_last_division
					elif 'cancer' in opr:
						return cll.time_since_cancer_in_neighbourhood
					else:
						return t
				else:
					if 'age' in opr:
						return cll.age
					if 'oxygen' in opr:
						o2_field = fields['oxygen'].vector().get_local()
						max_o2 = max(o2_field)
						return o2_field[cll.location]/max_o2
					if 'glucose' in opr:
						glu_field = fields['glucose'].vector().get_local()
						max_glu = max(glu_field)
						return glu_field[cll.location]/max_glu
					if 'lactate' in opr:
						lactate_field = fields['lactate'].vector().get_local()
						max_lact = max(lactate_field)
						if lactate_field[cll.location] < 0:
							lactate_field[cll.location] = 0
						return lactate_field[cll.location]/max_lact

					if 'local' in opr:
						if 'cancer' in opr:
							which_cell = 'cancer'
						elif 'fibroblasts' in opr:
							which_cell = 'fibroblast'
						else:
							raise ValueError('unrecognised type of cell')
						return ALISON.get_local(cell_population, which_cell, cll, neighbours[cll.location])


	@staticmethod
	def get_local(cell_pop, which_cell, cell, neighbours, distance=100): # the range of paracrine signals has been estimated to 100 um (Handly et al 2015) #TODO how big is an element?
		out = 0
		for c in cell_pop:
			if which_cell in c.type:
				if ALISON.does_it_count(c):
					loc = c.location
					if loc in neighbours:
						out += 1
		return out/len(neighbours)

	@staticmethod
	def does_it_count(cell):
		if 'fibroblasts' in cell.type:
			if cell.status == 3: #CAF
				return 1
			else:
				return 0
		elif 'cancer' in cell.type:
			if cell.status == 3: # proliferative status
				return 1
			else:
				return 0
		else:
			raise ValueError('you should not be here')

	@staticmethod
	# function that executes one operation in the probability string
	# TODO: here the assumption is that the operation has 2 operands. Evaluate the extension to 1 operator operations (exp, log)
	def execute_operation(o1, o2, op, idxs, cll, t, prob, pars, fields, cell_population, neighbours):
		vo1 = ALISON.get_operator_value(o1, pars, cll, t, fields, cell_population, neighbours)
		vo2 = ALISON.get_operator_value(o2, pars, cll, t, fields, cell_population, neighbours)
		if vo1<0:
			print('s')
		if vo2 <0:
			print('g')
		if op == '*':
			result = vo1 * vo2
		elif op == '/':
			result = vo1 / vo2
		elif op == '+':
			result = vo1 + vo2
		elif op == '-':
			result = vo1 - vo2
		else:
			raise ValueError('operation not recognized')
		before = prob[0: idxs[0]]
		after = prob[idxs[1] + 1:]
		out_prob = before + str(result) + after
		return out_prob

	@staticmethod
	def get_iterations(experiment_configuration):
		duration_hours, unit_duration = utility.convert_in_hours(experiment_configuration['duration'])
		resolution_hours, unit_resolution = utility.convert_in_hours(experiment_configuration['resolution'])
		iterations = int(duration_hours / resolution_hours)
		return iterations, resolution_hours, unit_resolution

	@staticmethod
	def initialize_outputs(initial_cell_population, fields, base_name):
		now = datetime.datetime.now()
		folder_name = os.getcwd() + os.path.sep + 'outputs' + os.path.sep + now.strftime("%d%m%Y_%H:%M:%S") \
					  + '_current_simulation_' + base_name + os.path.sep
		os.mkdir(folder_name)
		file_name = now.strftime("%d_%m_%Y_%H:%M:%S") + '_current_simulation_' + base_name + '_T =-1.pickle'
		fields_to_save = {}
		for f in fields:
			fields_to_save[f] = fields[f].vector().get_local()
		with open(folder_name + file_name, 'wb') as f:
			pickle.dump([fields_to_save, initial_cell_population], f)
		return folder_name

	@staticmethod
	# function that adds a cell to the matrix
	def add_cell(cell_configuration, cell_type, mesh, status, suitable_elements, number):
		out = []
		idxs = np.random.randint(0, len(suitable_elements), int(number))
		for i in idxs:
			position = suitable_elements[i]
			if 'cancer' in cell_type:  # do I need a smarter way to sort between cell types?
				out.append(cells.CancerCell(position, cell_configuration[cell_type], status))
			elif 'fibroblasts' in cell_type:
				out.append(cells.Fibroblast(position, cell_configuration[cell_type], status))
			elif 'mesothelial' in cell_type:
				out.append(cells.MesothelialCell(position, cell_configuration[cell_type], status))
			elif 'dummy' in cell_type:
				out.append(cells.CancerCell(position, cell_configuration[cell_type], status))
			else:
				raise ValueError('unrecognized cell type')
		return out

	@staticmethod
	def add_cells(cells, mesh_configuration, mesh):
		# function that adds the cells to the matrix.
		out = []
		for c in cells:
			n_cells = utility.engineering_notation(cells[c]['initial_condition'][0])
			status = cells[c]['initial_condition'][1].strip()
			layer_name = cells[c]['initial_condition'][2].strip()
			suitable_nodes = ALISON.get_suitable_elements(mesh, mesh_configuration[layer_name])
			out += ALISON.add_cell(cells, c, mesh, status, suitable_nodes, n_cells)
		# if you want to have more cells of the same type (cancer) but in different states (proliferant, quiescent) just
		# add separate lines in the configuration file
		return out

	@staticmethod
	def get_key_values(row, cell_type):
		r = row.split('\n')[0]
		if '->' in r:  # either behaviour or environmental interaction
			if ',' in r:  # behaviour
				start = int(r.split('->')[0])
				if '+' in r.split('->')[1].split(',')[0]:
					end = [int(x) for x in r.split('->')[1].split(',')[0].split('+')]
				else:
					end = int(r.split('->')[1].split(',')[0])
				probability = r.split(',')[1].strip()
				probability = probability.replace(" ", "")
				return 'dummy', {'start': start, 'end': end, 'probability': probability}

			else:  # environmental interaction
				cell = int(r.split('->')[0])
				variable = r.split('->')[1].split('_')[0].strip()
				action = r.split('_')[1].split('(')[0].strip()
				rate = r.split('(')[1].split(')')[0]
				return 'dummy', {'cell': cell, 'variable': variable, 'action': action, 'rate': rate}

		else:  # states or parameters
			if ':' in r:  # parameters
				parameter_name = r.split(':')[0]
				parameter_value = float(r.split(':')[1])
				return parameter_name, parameter_value

			else:  # cell states
				id_cell = int(r.split('.')[0])
				cell_name = r.split('.')[1].strip()
				return id_cell, cell_name

	@staticmethod
	def get_suitable_elements(mesh, layer):
		suitable_elements = []
		height_mesh = max(mesh.coordinates()[:, -1]) - min(mesh.coordinates()[:, -1])
		bounds = [min(mesh.coordinates()[:, -1]) + ((float(x) / 100) * height_mesh) for x in layer]
		for nn, n in enumerate(mesh.coordinates()):
			if bounds[0] < n[-1] < bounds[1]:
				suitable_elements.append(nn)
		return suitable_elements

	@staticmethod
	def read_cell_configuration(file_name):
		# Function that reads the cell configuration file #TODO evaluate how to change it when reading from protocol.
		sections = {'states', 'behaviour', 'environment interaction', 'parameters'}
		output_variable = {}
		cell_type = file_name.split('/')[-1].split('_')[0]
		with open(file_name) as f:
			for r in f.readlines():
				if r.split(':')[0] in sections:
					current_variable = r.split(':')[0]
					output_variable[current_variable] = {}
				else:
					if len(r) > 1 and 'configuration file' not in r:
						k, v = ALISON.get_key_values(r, cell_type)
						if k == 'dummy':
							k = utility.get_new_key(output_variable[current_variable])
						output_variable[current_variable][k] = v
		return output_variable

	@staticmethod
	def read_experimental_model(file_name):
		output_variable = {}
		with open(file_name) as F:
			for r in F.readlines():
				if not r.startswith('#'):
					if ':' in r:
						section = r.split(':')[0]
						output_variable[section] = {}
					elif '->' in r:
						key = r.split('->')[0].strip()
						value = r.split('->')[1].split('\n')[0].strip()
						if '.txt' in value:
							flag_text = 1
						else:
							flag_text = 0
						if ',' in value:
							value = [x.strip() for x in value.split(',')]
						if flag_text:
							output_variable[section][key] = {}
							new_value = []
							new_key = 'initial_condition'
							for v in value:
								if '.txt' in v:
									file_name = v
									if section == 'cells':
										full_path = os.getcwd() + os.path.sep + 'cell_types_configuration' + os.path.sep \
													+ file_name
										dummy = ALISON.read_cell_configuration(full_path)
										for t in dummy:
											output_variable[section][key][t] = dummy[t]
								else:
									new_value.append(v)

							output_variable[section][key][new_key] = new_value
						else:
							output_variable[section][key] = value
		return output_variable

	@staticmethod
	def read_main_configuration_file(c_file):
		# loading of the general configuration file. Here section headings have the ":" while fields have "->".
		# Multiple values associated with the sme field are separated by a comma.
		# #TODO substitute it with the reading of the protocol
		out_experiment = {}
		out_structure = {}
		out_cells = {}
		with open(c_file) as F:
			for ir, r in enumerate(F.readlines()):
				if not r.startswith('#'):
					if ':' in r:
						section = r.split(':')[0].strip()
						continue
					if '->' in r:
						key = r.split('->')[0].strip()
						value = r.split('->')[1].split('\n')[0].strip()
						if '.txt' in value:
							flag_file = 1
						else:
							flag_file = 0
						if ',' in value:
							value = [x.strip() for x in value.split(',')]
						if flag_file:
							value_sub_file = ALISON.read_sub_file(value, key)
							for v in value_sub_file:
								if v == 'mesh' or v == 'layers' or 'boundary' in v or 'material' in v:
									for v2 in value_sub_file[v]:
										out_structure[v2] = value_sub_file[v][v2]
								elif v == 'cells':
									for v2 in value_sub_file[v]:
										out_cells[v2] = value_sub_file[v][v2]
						else:
							if section == 'experimental conditions':
								out_experiment[key] = value
							else:
								raise ValueError('unexpected section value')
					else:
						if len(r) > 1:
							raise ImportError(
								'Unable to fin the correct separator within line ' + str(ir) + ' please refer'
																							   'to the documentation for more information')
		return out_experiment, out_structure, out_cells

	@staticmethod
	def read_sub_file(value, key):
		output_variable = {}
		if type(value) == str:
			file_name = value
		else:
			new_value = []
			for v in value:
				if '.txt' in v:
					file_name = v
				else:
					new_value.append(v)
		if key == 'experimental model':
			full_path = os.getcwd() + os.path.sep + 'experimental_models' + os.path.sep + file_name
			temp = ALISON.read_experimental_model(full_path)
			for t in temp:
				if t in output_variable:
					output_variable[t].update(temp[t])
				else:
					output_variable[t] = temp[t]
		elif 'cells' in file_name:
			full_path = os.getcwd() + os.path.sep + 'cell_types_configuration' + os.path.sep + file_name
			output_variable['cells'] = {}
			output_variable['cells'][key] = ALISON.read_cell_configuration(full_path)
			try:
				output_variable['cells'][key]['initial_condition'] = new_value
			except ValueError:
				raise ValueError('variable new_value undefined')

		else:
			raise ValueError('unrecognized file type')
		return output_variable
