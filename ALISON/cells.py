import random
import numpy as np


class Cell:
	def __init__(self, position):
		self.location = position
		self.age = 0
		self.update_status = 0
		self.time_death = None
		self.time_since_last_division = 0

	@staticmethod
	def is_drug_par(cnf, p):
		len_confs = len(cnf[p])
		if 'score' in cnf:
			ks = 'score'
		else:
			ks = 'scores'
		if 'drug' in cnf[ks]:
			len_drug = len(cnf[ks]['drug'])
			len_no_drug = len(cnf[ks]['no_drug'])
		else:
			drugs = list(cnf[ks].keys())  # it assumes the same number of parameters for each drug
			len_drug = len(cnf[ks][drugs[0]]['drug'])
			len_no_drug = len(cnf[ks][drugs[0]]['no_drug'])

		e_drug = np.absolute(len_drug - len_confs)
		e_nn_drug = np.absolute(len_no_drug - len_confs)
		if e_drug < e_nn_drug:
			return 1
		else:
			return 0

	@staticmethod
	def assign_parameters(config):
		parameters = list(config.keys())
		kscore = 'dummy'
		if 'score' in parameters:
			kscore = 'score'
			parameters.remove('score')
		if 'scores' in parameters:
			kscore = 'scores'
			parameters.remove('scores')
		if 'dummy' not in kscore:
			out = {}
			conf_id = Cell.pick_score(config[kscore])
			for p in parameters:
				if isinstance(config[p], list):
					if len(config[p]) == 2:
						out[p] = float(config[p][0])
					else:
						if Cell.is_drug_par(config, p):
							try:
								drug = p.split('_')[1]
								cid = conf_id['drug'][drug]
							except IndexError:
								drug = list(conf_id['drug'].keys())
								cid = conf_id['drug'][drug[0]]
						else:
							cid = conf_id['no_drug']
						out[p] = config[p][cid]
				else:
					out[p] = config[p]
			return out
		else:
			out = {}
			for p in parameters:
				if isinstance(config[p], list):
					out[p] = float(config[p][0])
				else:
					out[p] = config[p]
			return out

	'''
					if isinstance(config[p], list):
						has_str = 0
						for c in config[p]:
							if type(c) ==str:
								has_str =1
								break
						if not has_str:
							islist = 1
			if islist:
					if 'scores' in config:
							out = {}
							conf_id = Cell.pick_score(config['scores'])
							print(conf_id)
							for c in config:
									if isinstance(config[c], list):
											print(c, config[c])
											out[c]= config[c][conf_id]
									else:
											if c != 'scores':
													out[c] = config[c]
					else:
							out = {}
							for c in config:
									if isinstance(config[c], list):
											out[c] = float(config[c][0])
									else:
											out[c] = config[c]
					return out
			else:
					return config
		'''

	@staticmethod
	def pick_score(scr):
		out = {}
		if 'weights_drug' in scr:
			if sum(scr['weights_drug']) == 0:
				wgt_d = len(scr['weights_drug']) * [1]
			else:
				wgt_d = scr['weights_drug']

			if sum(scr['weight_no_drug']) == 0:
				wgt_nd = len(scr['weight_no_drug']) * [1]
			else:
				wgt_nd = scr['weight_no_drug']
			temp1 = random.choices(scr['no_drug'], weights=wgt_nd, k=1)
			temp2 = random.choices(scr['drug'], weights=wgt_d, k=1)
			out['drug'] = scr['drug'].index(temp2)
			out['no_drug'] = scr['no_drug'].index(temp1)
		else:
			drugs = list(scr.keys())
			if sum(scr[drugs[0]]['weight_no_drug']) == 0:
				wgt_nd = len(scr[drugs[0]]['weight_no_drug']) * [1]
			else:
				wgt_nd = scr[drugs[0]]['weight_no_drug']
			temp1 = random.choices(scr[drugs[0]]['no_drug'], weights=wgt_nd, k=1)
			out['no_drug'] = scr[drugs[0]]['no_drug'].index(temp1)
			out['drug'] = {}
			for d in drugs:
				if sum(scr[d]['weights_drug']) == 0:
					wgt_d = len(scr[d]['weights_drug']) * [1]
				else:
					wgt_d = scr[d]['weights_drug']
				temp1 = random.choices(scr[d]['drug'], weights=wgt_d, k=1)
				out['drug'][d] = scr[d]['drug'].index(temp1)

		return out

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
		out1 = {'behaviour': config['behaviour'], 'environment': config['environment interaction'],
				'current_rules': Cell.get_current_rules(config, status)}
		out2 = Cell.assign_parameters(config['parameters'])
		return out1, out2

	@staticmethod
	def get_status(config, stname):
		for s in config['states']:
			if config['states'][s] == stname:
				return s

	def double(self):
		self.time_since_last_division = 0
		self.age += 1
		self.update_status = 1

	def migrate(self, new_position):
		self.location = new_position
		self.time_since_last_division += 1
		self.age += 1
		self.update_status = 1

	def stay(self):
		self.time_since_last_division += 1
		self.age += 1
		self.update_status = 1

	def transition(self, status, t):
		self.status = status
		if self.configuration['states'][status] == 'dead':
			self.time_death = t
		self.rules['current_rules'] = self.update_current_rules(self.configuration, status)
		self.age += 1
		self.time_since_last_division += 1
		self.update_status = 1

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
		self.time_since_cancer_in_neighbourhood = 0
		self.configuration = configuration
		if type(status) == int:
			self.status = status
		else:
			self.status = self.get_status(configuration, status)
		self.rules, self.parameters = self.get_rules(configuration, self.status)


