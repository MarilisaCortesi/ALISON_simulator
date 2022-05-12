import numpy as np


def get_new_key(current_dict):
	# function that counts the number of keys in dictionary and returns the max+1
	if len(list(current_dict.keys())) == 0:
		return 0
	else:
		return max(list(current_dict.keys())) + 1


def engineering_notation(string_value):
	try:
		return float(string_value)
	except ValueError:
		if 'p' in string_value:
			return float(string_value.split('p')[0]) * 10 ** (-12)
		elif 'n' in string_value:
			return float(string_value.split('n')[0]) * 10 ** (-9)
		elif 'u' in string_value:
			return float(string_value.split('u')[0]) * 10 ** (-6)
		elif 'm' in string_value:
			return float(string_value.split('m')[0]) * 10 ** (-3)
		elif 'k' in string_value:
			return float(string_value.split('k')[0]) * 10 ** 3
		elif 'M' in string_value:
			return float(string_value.split('M')[0]) * 10 ** 6
		elif 'G' in string_value:
			return float(string_value.split('G')[0]) * 10 ** 9
		elif 'T' in string_value:
			return float(string_value.split('T')[0]) * 10 ** 12
		else:
			raise ValueError('unrecognized metric prefix')


def convert_in_hours(string):
	# allowed measurement units: s, min, h, day/days
	if 'day' in string:
		conversion_factor = 24
		duration = int(string.split('day')[0])
		unit = 'day'
	elif 'h' in string:
		conversion_factor = 1
		duration = int(string.split('h')[0])
		unit = 'hour'
	elif 'min' in string:
		conversion_factor = 60
		duration = int(string.split('min')[0])
		unit = 'minute'
	elif 's' in string:
		conversion_factor = 3600
		duration = int(string.split('s')[0])
		unit = 'second'
	else:
		raise ValueError('Unrecognized time unit. Please refer to the documentation for further details.')
	return duration * conversion_factor, unit


def convert_in_original_unit(amount, unit):
	if unit == 'day':
		return amount / 24
	elif unit == 'hour':
		return amount
	elif unit == 'minute':
		return amount * 60
	elif unit == 'second':
		return amount * 3600
	else:
		raise ValueError('Unrecognized measurement unit')


def update_measurement_unit(cell_population, unit):
	for c in cell_population:
		if unit == 'day':
			c.age /= 24
			c.time_since_last_division /= 24
			if c.time_death is not None:
				c.time_death /= 24
		elif unit == 'hour':
			continue
		elif unit == 'minute':
			c.age *= 60
			c.time_since_last_division *= 60
			if c.time_death is not None:
				c.time_death *= 60
		elif unit == 'seconds':
			c.age *= 3600
			c.time_since_last_division *= 3600
			if c.time_death is not None:
				c.time_death *= 3600
	return cell_population


def get_centroid(mesh, id_el):
	# function that computes the centroid of the specified element
	nodes_per_element = mesh.faces[0]
	idx_start = id_el * (nodes_per_element + 1)
	nodes = mesh.faces[idx_start + 1:idx_start + 1 + nodes_per_element]
	return np.mean(np.array(mesh.points[nodes, :]), axis=0)


def solve_equation(equation,cell):
	operations = ['*', '/', '+', '-'] # this might not work for complex eqs.
	for o in operations:
		if o in equation:
			temp = equation.split(o)
			if len(temp) > 2:
				raise ValueError('more than one operator operations not implemented')
			try:
				op1 = float(temp[0])
			except ValueError:
				op1 = cell.parameters[temp[0]]
			try:
				op2 = float(temp[1])
			except ValueError:
				op2 = cell.parameters[temp[1]]
			if o == '*':
				return op1 * op2
			elif o == '/':
				return op1/op2
			elif o == '+':
				return op1+op2
			else:
				return op1- op2
