class Cell:
	def __init__(self, position):
		self.location = position
		self.age = 0
		self.time_death = None
		self.time_since_last_division = 0

	@staticmethod
	def get_current_rules(config, status):
		out = {'behaviour': {}, 'environment': {}}
		for r in config['behaviour']:
			if config['behaviour'][r]['start'] == status:
				out['behaviour'][r] = config['behaviour'][r]

		for e in config['environment interaction']:
			if config['environment interaction'][e]['cell'] == status:
				out['environment'][e] = config['environment interaction'][e]
		return out

	@staticmethod
	def get_rules(config, status):
		out1 = {'behaviour': config['behaviour'], 'environment': config['environment interaction']}
		out1['current_rules'] = CancerCell.get_current_rules(config, status)
		out2 = config['parameters']
		return out1, out2

	@staticmethod
	def get_status(config, stname):
		for s in config['states']:
			if config['states'][s] == stname:
				return s



	def double(self):
		self.time_since_last_division = 0
		self.age += 1

	def migrate(self, new_position):
		self.location = new_position
		self.time_since_last_division += 1

	def stay(self):
		self.time_since_last_division += 1

	def transition(self, status, t):
		self.status = status
		if self.configuration['states'][status] == 'dead':
			self.time_death = t
		self.rules['current_rules'] = self.update_current_rules(self.configuration, status)

	@staticmethod
	def update_current_rules(config, status):
		out = {'behaviour': {}, 'environment': {}}
		for r in config['behaviour']:
			if config['behaviour'][r]['start'] == status:
				out['behaviour'][r] = config['behaviour'][r]

		for e in config['environment interaction']:
			if config['environment interaction'][e]['cell'] == status:
				out['environment'][e] = config['environment interaction'][e]
		return out


class CancerCell(Cell):
	def __init__(self, position, configuration, status):
		super().__init__(position)
		self.allowed_layers = 'all'
		self.type = 'cancer'
		self.configuration = configuration
		if type(status) == int:
			self.status = status
		else:
			self.status = self.get_status(configuration, status)
		self.rules, self.parameters = self.get_rules(configuration, self.status)

class MesothelialCell(Cell):
	def __init__(self, position, configuration, status):
		super().__init__(position)
		self.allowed_layers = 'mesothelial'
		self.type = 'mesothelial'
		self.configuration = configuration
		if type(status) == int:
			self.status = status
		else:
			self.status = self.get_status(configuration, status)
		self.rules, self.parameters = self.get_rules(configuration, self.status)


class Fibroblast(Cell):
	def __init__(self, position, configuration, status):
		super().__init__(position)
		self.allowed_layers = 'fibroblasts'
		self.type = 'fibroblasts'
		self.time_since_cancer_in_neighbourhood = 0 #TODO: this hypothesizes that no cancer cells are in the matrix at T = 0. Evaluate adding a check to make sure
		self.configuration = configuration
		if type(status) == int:
			self.status = status
		else:
			self.status = self.get_status(configuration, status)
		self.rules, self.parameters = self.get_rules(configuration, self.status)

