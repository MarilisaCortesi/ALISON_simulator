import itertools
import os
import warnings
import shutil
import datetime
import pickle
import fenics
import tqdm
import string
import numpy as np
import numpy.random as random
from ALISON import utility
from ALISON import DS3FE
from ALISON import cells


# TODO combination therapy,
# TODO calibration with clinical trial data
class ALISON:
	def __init__(self, configuration_file):
		self.base_name = configuration_file.split('.txt')[0]
		full_path_configuration = os.getcwd() + os.path.sep + 'experiment_configuration_files' + os.path.sep \
								  + self.base_name + '.txt'
		self.experiment_configuration, self.structure_configuration, self.cells_configuration = \
			self.read_main_configuration_file(full_path_configuration)
		self.update_scaling()
		self.update_timescale()
		self.mesh = DS3FE.initialise_mesh(self.structure_configuration['name'])
		if 'none' not in self.experiment_configuration['treatment']:
			self.experiment_configuration['treatment']['mesh elements'] = self.mesh.num_cells()

		neighbours_file_name = os.getcwd() + os.path.sep + 'meshes' + os.path.sep + \
							   self.structure_configuration['name'].split('.')[0] + '_neighbouring_nodes.pkl'
		# if os.path.isfile(neighbours_file_name):
		#       with open(neighbours_file_name, 'rb') as F:
		#               self.neighbours = pickle.load(F)
		# elsepython organotypic_model_main.py p42_cisplatin_IC50.txt:

		self.neighbours = self.get_neighbours(
			self.mesh.coordinates())  # , self.structure_configuration['max_distance'])
		# with open(neighbours_file_name, 'wb') as F:
		# print('load neighbours file')
		# pickle.dump(self.neighbours, F)
		self.function_space = fenics.FunctionSpace(self.mesh, 'P', 1)

		self.cell_population = self.add_cells(self.cells_configuration, self.structure_configuration, self.mesh)
		self.initial_conditions, self.fixed_flux = DS3FE.get_initial_conditions(self.experiment_configuration,
																				float(self.structure_configuration[
																						  'scale factor']),
																				self.mesh.num_cells())
		# print(self.initial_conditions)

		self.boundary_conditions = DS3FE.set_boundary_conditions(self.initial_conditions, self.function_space)
		self.f = DS3FE.initialise_f(self.mesh, self.initial_conditions, self.cell_population,
									self.function_space)
		self.fields = DS3FE.initialise_fields(self.initial_conditions, self.function_space)


	def update_timescale(self):
		t_res = int(self.experiment_configuration['step'].split('h')[0])
		self.structure_configuration['k'] = str(float(self.structure_configuration['k'])*t_res) #Thermal conductivity
		'''
		for c in self.cells_configuration:
			rates = []
			for e in self.cells_configuration[c]['environment interaction']:
				rt = self.cells_configuration[c]['environment interaction'][e]['rate']
				if '*' in rt:
					rt = rt.split('*')[1]
				if rt not  in rates:
					rates.append(rt)
			for r in rates:
				if type(self.cells_configuration[c]['parameters'][r]) == list:
					self.cells_configuration[c]['parameters'][r][0] = self.cells_configuration[c]['parameters'][r][0]*t_res
					self.cells_configuration[c]['parameters'][r][1] = 'timestep'
				elif type(self.cells_configuration[c]['parameters'][r]) == float:
					self.cells_configuration[c]['parameters'][r] = self.cells_configuration[c]['parameters'][r] * t_res
				else:
					raise ValueError('unrecognised parameter value')
			'''


	def update_scaling(self):
		scale_factor = float(self.structure_configuration['scale factor'])
		for c in self.cells_configuration:
			self.cells_configuration[c]['initial_condition'][0] = str(
				float(self.cells_configuration[c]['initial_condition'][0]) / scale_factor)
		volume_media = float(self.experiment_configuration['media'][1].split(' ')[0])
		unit = self.experiment_configuration['media'][1].split(' ')[1]
		scaled_volume = volume_media / scale_factor
		self.experiment_configuration['media'][1] = str(scaled_volume) + ' ' + unit

	@staticmethod
	def get_neighbours(coordinates):
		print('computing neighbouring nodes')
		out = {}
		max_dist = 0.1  # mm/h Liu 2021 (doi:10.1096/fj.202000101RR) the mesh is in mm
		for c in tqdm.tqdm(range(len(coordinates[:, 0]))):
			out[c] = {}
			difference = coordinates - coordinates[c, :]
			out[c]['distance'] = np.linalg.norm(difference, axis=1)
			out[c]['cell_range'] = np.where(out[c]['distance'] <= max_dist)[0]
		return out

	@staticmethod
	def reset_update_status(cell_population):
		for c in cell_population:
			c.update_status = 0
		return cell_population

	@staticmethod
	def pick_one(cell_pop):
		idx = random.randint(len(cell_pop))
		if cell_pop[idx].update_status == 0:
			return idx
		else:
			counter = 0
			while cell_pop[idx].update_status == 1:
				idx = random.randint(len(cell_pop))
				counter += 1
				if counter == 10:
					for cc, c in enumerate(cell_pop):
						if c.update_status == 0:
							idx = cc
							break
					break
			return idx

	def simulate(self, out_name, out_folder=os.getcwd()):
		# function that runs the simulation.
		simulation_folder = self.initialise_outputs(self.cell_population, self.fields, self.base_name, out_folder)
		trial_function = fenics.TrialFunction(self.function_space)
		test_function = fenics.TestFunction(self.function_space)
		diffusion_coefficient = fenics.Constant(
			float(self.structure_configuration['k']) / (float(self.structure_configuration['cv'])
														* float(self.structure_configuration['rho'])))
		dx = fenics.Measure('dx')
		F = {}
		a = {}
		L = {}
		for f in self.fields:
			F[f] = trial_function * test_function * dx + diffusion_coefficient * fenics.dot(
				fenics.grad(trial_function),
				fenics.grad(test_function)) * dx - (
						   self.fields[f] + diffusion_coefficient * self.f[f]) * test_function * dx
			a[f], L[f] = fenics.lhs(F[f]), fenics.rhs(F[f])
		t = 0
		step = int(self.experiment_configuration['step'].split(' ')[0])
		total_duration = int(self.experiment_configuration['duration'].split(' ')[0])
		iterations = int(total_duration/step)
		for n in range(iterations):
			t +=step
			trial_function_e = {}
			for v in F:
				trial_function = fenics.Function(self.function_space)
				fenics.solve(a[v] == L[v], trial_function, self.boundary_conditions[v])
				trial_function_e[v] = fenics.interpolate(trial_function, self.function_space)
				self.fields[v].assign(trial_function)
			ALISON.reset_update_status(self.cell_population)
			for oo in tqdm.tqdm(range(len(self.cell_population))):
				idx_cell = ALISON.pick_one(self.cell_population)
				o = self.cell_population[idx_cell]
				for r in range(step):
					if o.type == 'fibroblasts':
						check_neighbourhood = self.get_local(self.cell_population, o, 'cancer', 'proliferative',
														 self.neighbours)
						if check_neighbourhood > 0:
							o.time_since_cancer_in_neighbourhood += 1
						else:
							if o.time_since_cancer_in_neighbourhood > 0:
								o.time_since_cancer_in_neighbourhood -= 1  # if all the cancer cells are
					# gone from the fibroblast's neighbourhood its likelihood of becoming an activated CAF drops

					probability_vector = self.get_probabilities(o, t, self.fields, self.neighbours, self.cell_population,
															self.experiment_configuration['treatment'], total_duration)

					to_execute = self.choose_rule(probability_vector)
					log, self.cell_population = self.execute_rule(o, to_execute, self.mesh, self.initial_conditions, self.neighbours, t, step-r,
										self.cell_population, self.fields)

					#if o.type =='cancer':
					#	print(o.type,o.status, probability_vector, log)
					if o.update_status == 0:
						print(log)
						raise ValueError('something wrong with the update')
					if log['is_new_cell']:
						self.cell_population.append(log['new_cell'])
					if log['executed_rule'] == 'degradation':
						self.cell_population.pop(idx_cell)

			self.update_tracking_variables(simulation_folder, self.base_name, t,
										   self.fields,
										   self.cell_population)

			self.f = DS3FE.update_f(self.fields, self.cell_population, self.function_space)
			F = {}
			a = {}
			L = {}
			for f in self.fields:
				trial_function = fenics.TrialFunction(self.function_space)
				F[f] = trial_function * test_function * dx + diffusion_coefficient * fenics.dot(
					fenics.grad(trial_function),
					fenics.grad(test_function)) * dx - (
							   self.fields[f] + diffusion_coefficient * self.f[f]) * test_function * dx
				a[f], L[f] = fenics.lhs(F[f]), fenics.rhs(F[f])
			# trial_function_e = fenics.interpolate(trial_function, self.function_space)
			# self.fields[f].assign(trial_function_e)
		file_out = self.save_output(simulation_folder, out_name)
		return file_out

	@staticmethod
	def find_new_index(old_pop, new_pop, idx):
		position = old_pop[idx].location
		for nn, n in enumerate(new_pop):
			if n.location == position:
				return nn

	@staticmethod
	def update_cells_position(mesh, cls):
		for ic, c in enumerate(mesh['cells']):
			is_cell, side_effect = ALISON.is_occupied(cls, ic)
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
		side_effect = 'none'
		for c in cells:
			if c.location == location:
				if c.type == 'cancer':
					out_variable = 1
					break
				elif c.type == 'mesothelial':
					side_effect = 'clear'
					break
				elif c.type == 'fibroblasts':
					side_effect = 'move'
					break
				else:
					raise ValueError('unrecognised cell type')

		return out_variable, side_effect

	@staticmethod
	def save_output(output_folder, out_name):
		files = os.listdir(output_folder)
		complete_simulation = {}
		for f in files:
			time = int(float(f.split('=')[1].split('.pickle')[0]))
			if time < 0:
				time = 'initial_condition'
			with open(output_folder + os.path.sep + f, 'rb') as F:
				mesh, cell_population = pickle.load(F)
			complete_simulation[time] = {'mesh': mesh, 'cell_population': cell_population}
		file_name = output_folder.split('_current')[0] + '_' + out_name + '_complete_simulation.pickle'
		with open(file_name, 'wb') as F:
			pickle.dump(complete_simulation, F)
		shutil.rmtree(output_folder, ignore_errors=True)
		return file_name

	@staticmethod
	# function that looks for an empty neighbour of the current element.
	def find_empty_neighbour(mesh, initial_condition, neighbours, current_location, cell_population, fields,
							 no_side_effects=False):
		neighbouring_cells = neighbours[current_location]['cell_range']
		empty_neighbours = []
		scores = []
		side_effects = []
		for n in neighbouring_cells:
			occupied, side_effect = ALISON.is_occupied(cell_population, n)
			if not occupied:
				if no_side_effects:
					if side_effect == 'none':
						scores.append(ALISON.get_score(fields, initial_condition, n, mesh.coordinates()))
						empty_neighbours.append(n)
						side_effects.append(side_effect)
				else:
					scr = ALISON.get_score(fields, initial_condition, n, mesh.coordinates())
					if side_effects != 'none':
						bias = 0.1 * scr  # moving in an occupied position costs energy
					else:
						bias = 0
					scores.append(scr - bias)
					empty_neighbours.append(n)
					side_effects.append(side_effect)
		if len(empty_neighbours) > 0:
			max_score = max(scores)
			idxs_max = np.where(scores == max_score)[0]  # [ii for ii, i in enumerate(scores) if i == max_score]
			if len(idxs_max) > 1:
				x = np.random.randint(0, len(idxs_max))
			else:
				x = idxs_max[0]
			return empty_neighbours[x], side_effects[x]
		else:
			return -1, -1

	@staticmethod
	def get_score(fields, initial_condition, element, coords):
		glucose = fields['glucose'].vector().get_local()
		oxygen = fields['oxygen'].vector().get_local()
		lactate = fields['lactate'].vector().get_local()
		s_glucose = np.absolute((glucose[element] - initial_condition['glucose']) / initial_condition['glucose'])
		s_oxygen = np.absolute((oxygen[element] - initial_condition['oxygen']) / initial_condition['oxygen'])
		s_lactate = (lactate[element] - initial_condition['lactate']) / max(lactate)
		s_env = s_glucose + s_oxygen + (1 - s_lactate)
		z_el = coords[element][-1]
		min_z = min(coords[:, -1])
		max_z = max(coords[:, -1])
		s_pos = (max_z - z_el) / (max_z - min_z)
		return (2 * s_pos + s_env) / 2

	@staticmethod
	# function that executes the chosen rule.
	def execute_rule(cell, te, mesh, initial_condition, neighbours, iteration, substep, cell_population, fields):
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
				new_location, side_effect = ALISON.find_empty_neighbour(mesh, initial_condition, neighbours,
																		cell.location, cell_population, fields)
				cell.double()
				if side_effect != 'none':
					cell_population = ALISON.execute_side_effect(cell_population, new_location, side_effect, mesh,
																 initial_condition, neighbours, fields )
					#TODO does it have an effect? Check also 326

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
				new_location, side_effect = ALISON.find_empty_neighbour(mesh, initial_condition, neighbours,
																		cell.location, cell_population, fields)

				if side_effect != 'none':
					cell_population = ALISON.execute_side_effect(cell_population, new_location, side_effect, mesh,
																 initial_condition, neighbours, fields)
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
					cell.update_status = 1
				else:
					output_variable['executed_rule'] = 'transition_to_other_state'
					output_variable['is_new_cell'] = 0
					cell.transition(result, iteration)
		return output_variable, cell_population

	@staticmethod
	def execute_side_effect(cell_pop, loc, what, mesh, initial_conditions, neighbours, fields):
		for c in cell_pop:
			if c.location == loc:
				if what == 'clear':
					cell_pop.remove(c)
				elif what == 'move':
					new_location, _ = ALISON.find_empty_neighbour(mesh, initial_conditions, neighbours, c.location,
																  cell_pop, fields, no_side_effects=True)
					if new_location == -1:
						for n in neighbours[c.location]['cell_range']:
							new_location, _ = ALISON.find_empty_neighbour(mesh, initial_conditions, neighbours, n,
																		  cell_pop, fields, no_side_effects=True)
							if new_location != -1:
								break
					if new_location == -1:
						cell_pop.remove(c)
					else:
						c.migrate(new_location)
				else:
					raise ValueError('unrecognised side effect')
		return cell_pop

	@staticmethod
	def update_tracking_variables(simulation_folder, base_name, time, fields, cell_population):
		now = datetime.datetime.now()
		file_name = now.strftime("%d_%m_%Y_%H:%M:%S") + '_current_simulation_' + base_name + '_T =' + str(
			time) + '.pickle'
		fields_out = {}
		for f in fields:
			fields_out[f] = fields[f].vector().get_local()
		cpop1 = 0
		cpop2 = 0
		mpop = 0
		fpop1 = 0
		fpop2 = 0
		for c in cell_population:
			if c.type == 'mesothelial':
				if c.status == 2:
					mpop += 1
			if c.type == 'fibroblast':
				if c.status == 2:
					fpop1 += 1
				elif c.status == 3:
					fpop2 += 1
			if c.type == 'cancer':
				if c.status == 2:
					cpop1 += 1
				elif c.status == 3:
					cpop2 += 1
		#print(time, cpop1, cpop2, mpop, fpop1, fpop2)
		with open(simulation_folder + file_name, 'wb') as f:
			pickle.dump([fields_out, cell_population], f)

	@staticmethod
	# function that chooses which rule to execute.
	def choose_rule(probability_vector):
		cumulative = np.cumsum(probability_vector)
		if cumulative[-1] - 1 > 1e-6:
			raise ValueError('sum of probabilities above 1')
		rd_prob = np.random.random()
		tmp = cumulative - rd_prob < 0
		index = np.where(tmp == False)[0][0]
		return index

	@staticmethod
	def check_eligibility(neighbours, population, cll_type):
		out = False
		for n in neighbours['cell_range']:
			occupied, side_effect = ALISON.is_occupied(population, n)
			if not occupied:
				if cll_type == 'cancer':
					out = True
					break
				else:
					if side_effect == 'none':
						out = True
						break
		return out

	@staticmethod
	def get_probabilities(cll, t, fields, neighbours, cell_population, drug_characteristics, iterations):
		# function that gets the probability of a rule
		out = []
		#print('actual time', t)
		for r in sorted(cll.rules['current_rules']['behaviour']):
			if type(cll.rules['current_rules']['behaviour'][r]['end']) == list:
				eligible = ALISON.check_eligibility(neighbours[cll.location], cell_population, cll.type)
				if eligible:
					value = ALISON.get_value(cll, r, t, fields, cell_population, neighbours, drug_characteristics,
											 iterations)
				else:
					value = 0
			else:
				value = ALISON.get_value(cll, r, t, fields, cell_population, neighbours, drug_characteristics,
										 iterations)
				#print('aa',cll.type, cll.rules['current_rules']['behaviour'][r], value)
			if value < 0:
				value = 0.0
			out.append(value)
		if sum(out) > 1:
			out = [x / sum(out) for x in out]
			out.append(0)
		else:
			out.append(1 - sum(out))  # probability of remaining in the same state
		# print(cll.status, out)
		return out

	@staticmethod
	def adjust_probability_string(pstring):
		if '-' in pstring:
			temp = pstring.split('-')
			new_minus = '~'
			out = new_minus.join([str(elem) for elem in temp])
			return out
		else:
			return pstring

	@staticmethod
	def get_value(cll, r, t, fields, cell_population, neighbours, drug_characteristics, iterations):
		# another function for the interpretation of the probability strings
		probability_string = ALISON.adjust_probability_string(cll.rules['current_rules']['behaviour'][r]['probability'])
		parameter_values = cll.parameters
		operators_types = ['*', '/', '~', '+']
		n_ops = 0
		for p in probability_string:
			if p in operators_types:
				n_ops += 1
		exec_op = 0
		while exec_op < n_ops:
			next_op, op1, op2, idxs = ALISON.find_next_operation(operators_types, probability_string)
			value_op1 = ALISON.get_operator_value(op1, parameter_values, cll, t, fields, cell_population,
												  neighbours, drug_characteristics, iterations)
			value_op2 = ALISON.get_operator_value(op2, parameter_values, cll, t, fields, cell_population,
												  neighbours, drug_characteristics, iterations)
			probability_string = ALISON.execute_operation(value_op1, value_op2, next_op, idxs, probability_string)
			exec_op += 1

		# print(probability_string)
		return float(probability_string)

	@staticmethod
	# function that gets the order of cell addition to the mesh
	def get_order(population):
		order = np.random.sample(population, len(population))
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
				if op_pos == 0:
					raise ValueError('operator as first element')
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
	def get_operator_value(opr, pars, cll, t, fields, cell_population, neighbours, drug_characteristics, iterations):
		if opr in pars:
			return pars[opr]
		else:
			try:
				return float(opr)
			except ValueError:
				if 'time' in opr:  # time, time_since_death, time_since_last_division
					if 'death' in opr:
						if cll.time_death is None:
							raise ValueError("This cell is not dead")
						else:
							return (
										t - cll.time_death) / 24  # dead cells are cleared quickly in the tissues Yoon 2017 10.5483/BMBRep.2017.50.10.147
					elif 'division' in opr:
						return cll.time_since_last_division / 100  # most cells have a doubling time below tht
					# elif 'cancer' in opr:
					#        return cll.time_since_cancer_in_neighbourhood/iterations

					else:
						return t / iterations
				else:
					if 'age' in opr:
						return cll.age / 2500  # Hayflick limit
					if 'drug' in opr:
						try:
							drug_type = opr.split('_')[1]
							drug_field = fields[drug_type].vector().get_local()
						except IndexError:
							drug_type = 'drug'
							drug_field = fields[drug_type].vector().get_local()
						p_effect = utility.sigmoid(drug_field[cll.location], drug_characteristics, drug_type)
						# print('drug', p_effect)
						return p_effect
					if 'oxygen' in opr:
						o2_field = fields['oxygen'].vector().get_local()
						max_o2 = max(o2_field)
						#print(o2_field[cll.location]/max_o2, 'o2')
						if o2_field[cll.location] < 0:
							return 0
						else:
							return o2_field[cll.location] / max_o2
					if 'glucose' in opr:
						glu_field = fields['glucose'].vector().get_local()
						max_glu = max(glu_field)
						#print(glu_field[cll.location], max_glu, 'glu')
						if glu_field[cll.location] < 0:
							return 0
						else:
							return glu_field[cll.location] / max_glu
					if 'lactate' in opr:
						lactate_field = fields['lactate'].vector().get_local()
						max_lact = max(lactate_field)
						#print('lactate', lactate_field[cll.location])
						return lactate_field[cll.location] / max_lact

					if 'local' in opr:
						temp = opr.split('_')
						which_cell = temp[-1]
						if len(temp) == 3:
							which_status = temp[-2]
						elif len(temp) == 4:
							which_status = temp[1] + '_' + temp[2]
						else:
							raise ValueError('unrecognised cell status')
						return ALISON.get_local(cell_population, cll, which_cell, which_status, neighbours)

	@staticmethod
	def get_local(cell_pop, cell, which_cell, which_status, neighbours,
				  distance=0.1):  # the range of paracrine signals has been estimated to 100 um (Handly et al 2015)
		out = 0
		distance_from_cell = neighbours[cell.location]['distance']
		total = 0
		for c in cell_pop:
			if distance_from_cell[c.location] <= distance:
				total += 1
				if which_cell in c.type:
					if ALISON.does_it_count(c, which_status):
						out += 1
		if total>0:
			return out / total
		else:
			return 0

	@staticmethod
	def check_neighbour(n_id, cpop, whch_cll, which_status):
		ocpd = 0
		ctns = 0
		for c in cpop:
			if c.location == n_id:
				ocpd = 1
				if whch_cll == c.type:
					ctns = ALISON.does_it_count(c, which_status)
				break
		return ocpd, ctns

	@staticmethod
	def does_it_count(cell, sts):
		if cell.configuration['states'][cell.status] == sts:
			return 1
		else:
			return 0

	@staticmethod
	# function that executes one operation in the probability string
	# TODO: here the assumption is that the operation has 2 operands. Evaluate the extension to 1 operator operations (exp, log)
	def execute_operation(vo1, vo2, op, idxs, prob):
		if op == '*':
			result = vo1 * vo2
		elif op == '/':
			result = vo1 / vo2
		elif op == '+':
			result = vo1 + vo2
		elif op == '~':
			result = vo1 - vo2
		else:
			raise ValueError('operation not recognized')
		before = prob[0: idxs[0]]
		after = prob[idxs[1] + 1:]
		result_str = '{res:.9f}'
		# result = round(decimal.Decimal(result), 9)
		out_prob = before + result_str.format(res=result) + after
		return out_prob

	@staticmethod
	def get_iterations(experiment_configuration):
		duration_hours, unit_duration = utility.convert_in_hours(experiment_configuration['duration'])
		resolution_hours, unit_resolution = utility.convert_in_hours(experiment_configuration['resolution'])
		iterations = int(duration_hours / resolution_hours)
		return iterations, resolution_hours, unit_resolution

	@staticmethod
	def initialise_outputs(initial_cell_population, fields, base_name, base_folder):
		now = datetime.datetime.now()
		folder_name = base_folder + os.path.sep + 'outputs' + os.path.sep + now.strftime("%d%m%Y_%H:%M:%S") \
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
	def add_cell(cell_configuration, cell_type, status, suitable_elements, number):
		out = []
		if int(number) > len(suitable_elements):
			idxs = range(len(suitable_elements))
			print('more cells than spaces, filling all the available elements')
		else:
			if cell_type == 'cancer':
				idxs = range(int(number))
			else:
				idxs = []
				while len(idxs) < int(number):
					new_pos = np.random.randint(0, len(suitable_elements))
					if new_pos not in idxs:
						idxs.append(new_pos)
		for i in idxs:
			position = suitable_elements[i]
			if 'cancer' in cell_type:
				out.append(cells.CancerCell(position, cell_configuration[cell_type], status))
			elif 'fibroblasts' in cell_type:
				out.append(cells.Fibroblast(position, cell_configuration[cell_type], status))
			elif 'mesothelial' in cell_type:
				out.append(cells.MesothelialCell(position, cell_configuration[cell_type], status))
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
			suitable_nodes = ALISON.get_suitable_elements(mesh, mesh_configuration, layer_name)
			out += ALISON.add_cell(cells, c, status, suitable_nodes, n_cells)
		# if you want to have more cells of the same type (cancer) but in different states (proliferant, quiescent) just
		# add separate lines in the configuration file
		return out

	@staticmethod
	def get_key_values(row):
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
				try:
					parameter_value = float(r.split(':')[1].split('#')[0])
					if '#' in r:
						parameter_value = [parameter_value, r.split('*')[1]]
				except ValueError:
					if 'pkl' in r or 'pickle' in r:
						files = r.split(':')[1].split(',')
						scores_all = {}
						for f in files:
							try:
								full_path = os.getcwd() + os.path.sep + 'scores' + os.path.sep \
											+ f.strip()
								with open(full_path, 'rb') as F:
									scores_all[f.strip()] = pickle.load(F)

							except:
								raise FileExistsError(
									'There is something wrong with the score file. Check that it is in the scores folder and it is a pickle file.')
						parameter_value = scores_all

					else:
						parameter_value = 'expression: ' + r.split(':')[1].split('#')[0]
				return parameter_name, parameter_value

			else:  # cell states
				id_cell = int(r.split('.')[0])
				cell_name = r.split('.')[1].strip()
				return id_cell, cell_name

	@staticmethod
	def get_suitable_elements(mesh, configuration, layer):
		suitable_elements = []
		height_mesh = max(mesh.coordinates()[:, -1]) - min(mesh.coordinates()[:, -1])
		bounds = [min(mesh.coordinates()[:, -1]) + ((float(x) / 100) * height_mesh) for x in configuration[layer]]
		z_value = []
		for nn, n in enumerate(mesh.coordinates()):
			if bounds[0] < n[-1] < bounds[1]:
				suitable_elements.append(nn)
				z_value.append(n[-1])
		if 'top' in layer:
			suitable_elements = ALISON.sort_by_z(suitable_elements, z_value)
		return suitable_elements

	@staticmethod
	def sort_by_z(elems, zv):
		out = [x for _, x in sorted(zip(zv, elems))]
		return out

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
						k, v = ALISON.get_key_values(r)
						if isinstance(v, str) and 'expression' in v:
							v = utility.solve_equation(v.split('expression:')[1], output_variable['parameters'])
						if k == 'dummy':
							k = utility.get_new_key(output_variable[current_variable])
						if k == 'parameter file':
							if 'n_scores' in v.keys():  # single drug
								if v['n_scores'] != 2:
									raise ValueError('only a 2 score system is implemented')
								output_variable['parameters']['scores'] = {'no_drug': v['score_1'],
																		   'drug': v['score_2'],
																		   'weight_no_drug': v['weights_1'],
																		   'weights_drug': v['weight_2']}
								v = ALISON.assign_parameters(v)  # output_variable['parameters'])
								k = 'remove'
							else:
								output_variable['parameters']['score'] = {}
								for drg in v:
									if v[drg]['n_scores'] != 2:
										raise ValueError('only a 2 score system is implemented. Please modify drug',
														 drg)
									drug = drg.split('_')[2]
									output_variable['parameters']['score'][drug] = {'no_drug': v[drg]['score_1'],
																					'drug': v[drg]['score_2'],
																					'weight_no_drug': v[drg][
																						'weights_1'],
																					'weights_drug': v[drg]['weight_2']}

								v = ALISON.assign_parameters(v)  # output_variable['parameters'])
								k = 'remove'

						if k == 'remove':
							for p in v:
								output_variable[current_variable][p] = v[p]
						else:
							output_variable[current_variable][k] = v
		return output_variable

	@staticmethod
	def assign_parameters(dict_pars):
		if len(dict_pars) == 1:
			K = list(dict_pars.keys())
			dict_pars = dict_pars[K[0]]
		if 'confs_1' in dict_pars:
			confs_no_drug = dict_pars['confs_1']
			confs_drug = dict_pars['confs2']
			par_names_no_drug = list(confs_no_drug[0].keys())
			par_names_drug = list(confs_drug[0].keys())
			out = {}
			for p in par_names_no_drug:
				out[p] = []
				for c in sorted(confs_no_drug):
					out[p].append(confs_no_drug[c][p])

			for p in par_names_drug:
				out[p] = []
				for c in sorted(confs_drug):
					out[p].append(confs_drug[c][p])
		else:
			drgs = list(dict_pars.keys())
			confs_no_drug = dict_pars[drgs[0]]['confs_1']
			par_names_no_drug = list(confs_no_drug[0].keys())
			out = {}
			for p in par_names_no_drug:
				out[p] = []
				for c in sorted(confs_no_drug):
					out[p].append(confs_no_drug[c][p])
			for d in drgs:
				confs_drug = dict_pars[d]['confs2']
				par_names_drg = list(confs_drug[0].keys())
				drug_name = d.split('_')[2]
				for p in par_names_drg:
					new_name = p + '_' + drug_name
					out[new_name] = []
					for c in sorted(confs_drug):
						out[new_name].append(confs_drug[c][p])
		return out

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
					# print(r)
					if ':' in r:
						section = r.split(':')[0].strip()
						continue
					if '->' in r:
						key = r.split('->')[0].strip()
						value = r.split('->')[1].split('\n')[0].strip()
						if '.txt' in value:  # TODO it assumes only text files, is it worth including other?
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
								elif v == 'drug':
									out_experiment['treatment'] = value_sub_file
								else:
									raise ValueError('Unrecognised Value: ', v)
						else:
							if section == 'experimental conditions':
								out_experiment[key] = value
							else:
								raise ValueError('unexpected section value')
					else:
						if len(r) > 1:
							raise ImportError(
								'Unable to find the correct separator within line ' + str(ir) + ' please refer'
																								'to the documentation for more information')
		return out_experiment, out_structure, out_cells

	@staticmethod
	def read_drug_configuration(treatment, file_name):
		if isinstance(file_name, str):  # 1 drug
			full_path = os.getcwd() + os.path.sep + 'drugs' + os.path.sep + file_name
			drug = ALISON.load_drug_features(full_path)
			if 'M' in treatment[1]:  # concentration in molar
				value = utility.engineering_notation(treatment[1].split('M')[0])
				new_value = value * drug['molecular weight [g/mol]']  # g/l = ug/ul
				drug['dose'] = new_value
			else:
				raise ValueError('unrecognised unit measurement')
		else:
			drug = {}
			for ff, f in enumerate(file_name):
				drug_name = f.split('_')[0]
				full_path = os.getcwd() + os.path.sep + 'drugs' + os.path.sep + f
				drug[drug_name] = ALISON.load_drug_features(full_path)

				idx = 2 * ff + 1
				if 'M' in treatment[idx]:  # concentration in molar
					value = utility.engineering_notation(treatment[idx].split('M')[0])
					new_value = value  # * drug[drug_name]['molecular weight [g/mol]'] #g/l = ug/ul
					drug[drug_name]['dose'] = new_value
				else:
					raise ValueError('unrecognised unit measurement')
		return drug

	@staticmethod
	def load_drug_features(flnm):
		drg = flnm.split(os.path.sep)[-1].split('.txt')[0]
		out = {}
		out['name'] = drg
		with open(flnm, 'r') as F:
			for r in F.readlines():
				key = r.split(':')[0]
				value = float(r.split(':')[1].split('\n')[0])
				out[key] = value
		return out

	@staticmethod
	def read_sub_file(value, key):
		output_variable = {}
		if type(value) == str:
			file_name = value
		else:
			new_value = []
			file_name = []
			for v in value:
				if '.txt' in v:
					file_name.append(v)
				else:
					new_value.append(v)
			if len(file_name) == 1:
				file_name = file_name[0]
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
		elif key == 'treatment':
			output_variable['drug'] = ALISON.read_drug_configuration(value, file_name)
			if 'dose' in output_variable[
				'drug']:  # only for 1 drug. For combo treatments this is done in read_drug_configuration
				output_variable['drug']['dose'] = [utility.engineering_notation(i) for i in new_value]


		else:
			raise ValueError('unrecognized file type', file_name)
		return output_variable

