import os
import warnings
import pickle

import matplotlib.pyplot as plt
import numpy as np
import itertools


def load_cell_features(fnm):
	out = {}
	with open(fnm) as F:
		for rr, r in enumerate(F.readlines()):
			if len(r) > 1 and rr > 0:
				temp = r.split(':')
				if len(temp[1]) == 1:
					cell_line = temp[0]
					out[cell_line] = {}
				else:
					feature = temp[0]
					value = interpret_value_cells(feature, temp[1].split('\n')[0])
					out[cell_line][feature] = value
	return out


def calibrate_new_cell_lines(file_name):
	full_path = os.path.join(os.getcwd(), 'feature_files', file_name)
	cell_lines = load_cell_features(full_path)
	for c in cell_lines:
		scores_original_cell_lines = match_cell_lines2(cell_lines[c])


def match_cell_lines2(feats):
	cell_lines_features = load_cell_lines(os.path.join(os.getcwd(), 'feature_files', 'cell_lines.txt'))
	features_list = 4 * [-1]
	if feats['Known drug resistance'] == 0:
		features_list[1] = 0
	elif feats['Known drug resistance'] == -1:
		features_list[1] = -1
	else:
		which_drugs = feats['Known drug resistance'][1]
		if ',' in which_drugs:
			features_list[1] = 3
		elif which_drugs == 'platinum':
			features_list[1] = 2
		else:
			features_list[1] = 1
	if 'Recurrence' in feats:
		if feats['Recurrence'] == 0:
			features_list[0] = 0
		elif feats['Recurrence'] == -1:
			features_list[0] = -1
		else:
			features_list[0] = feats['Recurrence'][1]
	if 'BRCA status' in feats:
		if 'WT' in feats['BRCA status']:
			if 'reversion' in feats['BRCA status']:
				features_list[2] = 3
			else:
				features_list[2] = 0
		else:
			if '1' in feats['BRCA status']:
				features_list[2] = 1
			elif '2' in feats['BRCA status']:
				features_list[2] = 2
			else:
				raise ValueError('unrecognised BRCA status')
	if 'other mutations' in feats:
		if feats['other mutations'] == 0:
			features_list[3] = 0
		else:
			other_mutations = interpret_mutations(feats['other mutations'])
			features_list[3] = sum(other_mutations)  # TODO is this reasonable?
	'''
    deltas = {}
    total = 0
	for c in cell_lines_features:
		deltas[c] = 0
		nf = 0
		for ff, f in enumerate(features_list):
			cf = cell_lines_features[c][ff]
			if f > -1:
				if cf > -1:
					nf += 1
					delta_features = np.absolute(cf - f)
					deltas[c] += delta_features

		deltas[c] = 4*(deltas[c]/nf) # normalise on the number of features
		total += deltas[c]
	score = {}
	for s in deltas:
		score[s] = 1-(deltas[s]/total)
	return score
        '''


def interpret_value_cells(var, vl):
	if 'not known' in vl:
		return -1
	if 'Recurrence' in var or 'Known drug resistance' in var:
		if 'no' in vl:
			return 0
		else:
			if '(' in vl:
				specification = vl.split('(')[1].split(')')[0]
				if 'not known' in specification:
					return 1
				else:
					if 'Recurrence' in var:
						specification = int(specification)
					return [1, specification]
	elif var == 'doubling time [h]':
		return float(vl)
	else:
		if vl == 'no':
			return 0
		else:
			return vl


def calibrate_digital_twin(patient_id):
	print(patient_id)
	features = {}
	features_file = patient_id + '.txt'
	full_path = os.path.join(os.getcwd(), 'feature_files', features_file)
	with open(full_path, 'r') as F:
		for r in F.readlines():
			if '#' in r:
				continue
			if ':' in r:
				variable = r.split(':')[0]
				value = r.split(':')[1].split('\n')[0].strip()
				if value == 'not known':
					continue
				else:
					value = interpret_value(variable, value)
					features[variable] = value
	scores_cell_lines = match_cell_lines(features)
	bias = get_bias(features)
	print(bias)
	renormalise = 1
	if bias > 1:  # worst prognosis
		increase = bias - 1
		scores_cell_lines['PEO4'] += increase
		scores_cell_lines['OVCAR4'] += increase
		scores_cell_lines['OAW28'] += increase
		scores_cell_lines['PEO1'] -= increase
		if scores_cell_lines['PEO1'] < 0:
			scores_cell_lines['PEO1'] = 0
			renormalise = 1
		scores_cell_lines['CaOV3'] -= increase
		if scores_cell_lines['CaOV3'] < 0:
			scores_cell_lines['CaOV3'] = 0
			renormalise = 1
			scores_cell_lines['OVSAHO'] -= increase
		if scores_cell_lines['OVSAHO'] < 0:
			scores_cell_lines['OVSAHO'] = 0
			renormalise = 1
	else:
		decrease = -1 * bias
		scores_cell_lines['PEO4'] -= decrease
		if scores_cell_lines['PEO4'] < 0:
			scores_cell_lines['PEO4'] = 0
			renormalise = 1
		scores_cell_lines['OVCAR4'] -= decrease
		if scores_cell_lines['OVCAR4'] < 0:
			scores_cell_lines['OVCAR4'] = 0
			renormalise = 1
		scores_cell_lines['OAW28'] -= decrease
		if scores_cell_lines['OAW28'] < 0:
			scores_cell_lines['OAW28'] = 0
			renormalise = 1
		scores_cell_lines['PEO1'] += decrease
		scores_cell_lines['CaOV3'] += decrease
		scores_cell_lines['OVSAHO'] += decrease
	if renormalise:
		total = 0
		for s in scores_cell_lines:
			total += scores_cell_lines[s]
		for s in scores_cell_lines:
			scores_cell_lines[s] = scores_cell_lines[s] / total

	scores_cell_lines = percentage_score(scores_cell_lines)
	patient_distribution = get_patient_distribution(scores_cell_lines)
	putative_IC50, molecular_weights = get_putative_IC50(scores_cell_lines)
	for d in patient_distribution:
		file_score = os.path.join(os.getcwd(), 'scores', 'scores_' + patient_id + '_' + d + '_combined.pkl')
		with open(file_score, 'wb') as F:
			pickle.dump(patient_distribution[d], F)
	for d in molecular_weights:
		file_drug = os.path.join(os.getcwd(), 'drugs', d + '_' + patient_id + '.txt')
		with open(file_drug, 'w') as F:
			F.write(molecular_weights[d])
			to_write = 'IC50 [M]: ' + str(putative_IC50[d]) + '\n'
			F.write(to_write)


def get_putative_IC50(scrs):
	print(scrs.keys())
	folder = os.path.join(os.getcwd(), 'drugs')
	files = os.listdir(folder)
	ic50s = {}
	weights = {}
	MW = {}
	for f in files:
		drug = f.split('_')[0]
		cell_line = f.split('_')[1].split('.txt')[0]
		if cell_line not in scrs:
			continue
		if drug not in ic50s:
			ic50s[drug] = {}
		if cell_line not in weights:
			weights[cell_line] = scrs[cell_line]
		with open(os.path.join(folder, f), 'r') as F:
			for r in F.readlines():
				if 'IC50' in r:
					ic50s[drug][cell_line] = float(r.split('\n')[0].split(':')[1])
				if 'molecular' in r:
					MW[drug] = r
	out = {}
	for i in ic50s:
		ic_50_list = []
		weight_list = []
		for c in ic50s[i]:
			ic_50_list.append(ic50s[i][c])
			weight_list.append(weights[c])
		out[i] = np.average(ic_50_list, weights=weight_list)
	return out, MW


def percentage_score(scr):
	min_s = 100
	max_s = -100
	for s in scr:
		if scr[s] < min_s:
			min_s = scr[s]
		if scr[s] > max_s:
			max_s = scr[s]
	temp = {}
	total = 0
	for s in scr:
		temp[s] = scr[s] - min_s
		total += temp[s]
	out = {}
	for t in temp:
		out[t] = temp[t] / total
	return out


def get_patient_distribution(scr_cll):
	score_folder = os.path.join(os.getcwd(), 'scores')
	cells = list(scr_cll.keys())
	score_files = os.listdir(score_folder)
	distributions = {}
	for f in score_files:
		temp = f.split('.pkl')[0].split('_')
		cell = temp[1]
		if cell not in cells:
			continue
		if len(temp) == 2:
			treatment = 'no'
		else:
			treatment = temp[-2]
		if cell not in distributions:
			distributions[cell] = {}
		with open(os.path.join(score_folder, f), 'rb') as F:
			distributions[cell][treatment] = pickle.load(F)
	out = {}
	scores = ['score_1', 'weights_1', 'score_2', 'weight_2']
	other_keys = ['n_scores', 'confs_1', 'confs2']
	drugs = list(distributions[cells[0]].keys())
	for d in drugs:
		out[d] = {}
		for s in scores:
			out[d][s] = []
			if s in distributions[cells[0]][d]:
				len_vect = len(distributions[cells[0]][d][s])
			else:
				len_vect = len(distributions[cells[0]]['no'][s])

			for idx in range(len_vect):
				value = 0
				for c in scr_cll:
					if len(distributions[c][d][s]) < len_vect:
						delta = (len_vect - len(distributions[c][d][s]))
						val = distributions[c][d][s][-1]
						for dd in range(delta):
							distributions[c][d][s].append(val)
					if s in distributions[c][d]:
						value += scr_cll[c] * distributions[c][d][s][idx]
					else:
						value += scr_cll[c] * distributions[c]['no'][s][idx]
				out[d][s].append(value)
		for o in other_keys:
			out[d][o] = distributions[c][d][o]
	return out


def get_bias(feats):
	bias_age = 0
	bias_stage = 0
	bias_debulking = 0
	bias_CA125 = 0
	if 'Age' in feats:
		average_diagnosis = 63
		bias_age = (feats['Age'] - average_diagnosis) / average_diagnosis
	if 'Stage' in feats:
		max_stage = 4.3
		min_stage = 1
		y_max = 1
		y_min = -1
		slope = (y_max - y_min) / (max_stage - min_stage)
		intercept = y_min - slope * min_stage

		bias_stage = feats['Stage'] * slope + intercept
	if 'Interval debulking' in feats:
		if isinstance(feats['Interval debulking'], list):
			if 'R1' in feats['Interval debulking'][1]:
				bias_debulking = 0.5
			else:
				bias_debulking = 0
		else:
			if feats['Interval debulking'] == 0:
				bias_debulking = 1
			elif feats['Interval debulking'] == 1:
				bias_debulking = 0
		print('s')
	if 'CA125' in feats:
		normal_limit = 35
		max_value = 1000
		min_y = 0
		max_y = 1
		slope = (max_y - min_y) / (max_value - normal_limit)
		intercept = min_y - normal_limit * slope
		if feats['CA125'] > normal_limit:
			bias_CA125 = slope * feats['CA125'] + intercept
		else:
			bias_CA125 = 0
	return bias_age + bias_stage + bias_CA125 + bias_debulking


def load_cell_lines(path_file):
	out = {}
	with open(path_file, 'r') as F:
		for r in F.readlines():
			if ':' in r:
				temp = r.split('\n')[0].split(':')
				if len(temp[1]) == 0:
					cell_line = temp[0]
					out[cell_line] = 4 * [-1]
				else:
					if temp[0] == 'Recurrence':
						idx = 0
					elif temp[0] == 'Known drug resistance':
						idx = 1
					elif temp[0] == 'BRCA status':
						idx = 2
					elif temp[0] == 'other mutations':
						idx = 3
					else:
						raise ValueError('Unrecognised property')
				if 'no' in temp[1] and 'known' not in temp[1]:
					out[cell_line][idx] = 0
				elif 'yes' in temp[1]:
					value = temp[1].split('(')[1].split(')')[0]
					if value.isdigit():
						out[cell_line][idx] = float(value)
					else:
						if idx == 0:
							if value == 'not known':
								out[cell_line][idx] = 1
							else:
								raise ValueError('Recurrence type not implemented, ', value)
						elif idx == 1:
							if 'platinum' in value:
								out[cell_line][idx] = 2
							elif 'multidrug' in value:
								out[cell_line][idx] = 3
				elif 'WT' in temp[1]:
					if 'reversion' in temp[1]:
						out[cell_line][idx] = 3
					else:
						out[cell_line][idx] = 0
				elif 'BRCA' in temp[1]:
					if '1' in temp[1]:
						out[cell_line][idx] = 1
					else:
						out[cell_line][idx] = 2
				elif 'NF1' in temp[1]:
					out[cell_line][idx] = 3
	return out


def match_cell_lines(feats):
	cell_lines_features = load_cell_lines(os.path.join(os.getcwd(), 'feature_files', 'cell_lines.txt'))
	features_list = 4 * [-1]
	if feats['Known drug resistance'] == 0:
		features_list[1] = 0
	else:
		if 'platinum' in feats['Known drug resistance'][1]:
			features_list[1] = 2
		elif 'multidrug' in feats['Known drug resistance'][1]:
			features_list[1] = 3
		else:
			features_list[1] = 1
	if 'Recurrence' in feats:
		if feats['Recurrence'] == 0:
			features_list[0] = 0
		else:
			features_list[0] = feats['Recurrence'][1]
	if 'BRCA status' in feats:
		if 'WT' in feats['BRCA status']:
			if 'reversion' in feats['BRCA status']:
				features_list[2] = 3
			else:
				features_list[2] = 0
		else:
			if '1' in feats['BRCA status']:
				features_list[2] = 1
			elif '2' in feats['BRCA status']:
				features_list[2] = 2
			else:
				raise ValueError('unrecognised BRCA status')
	if 'other mutations' in feats:
		if feats['other mutations'] == 0:
			features_list[3] = 0
		else:
			other_mutations = interpret_mutations(feats['other mutations'])
			features_list[3] = sum(other_mutations)  # TODO is this reasonable?
	deltas = {}
	total = 0
	for c in cell_lines_features:
		deltas[c] = 0
		nf = 0
		for ff, f in enumerate(features_list):
			cf = cell_lines_features[c][ff]
			if f > -1:
				if cf > -1:
					nf += 1
					delta_features = np.absolute(cf - f)
					deltas[c] += delta_features

		deltas[c] = 4 * (deltas[c] / nf)  # normalise on the number of features
		total += deltas[c]
	score = {}
	for s in deltas:
		score[s] = 1 - (deltas[s] / total)
	return score


def interpret_mutations(muts):
	mutations = [m.strip() for m in muts.split(',')]
	out = []
	known_mutations = load_mutations(os.path.join(os.getcwd(), 'ALISON', 'mutations.txt'))
	print(known_mutations)
	for m in mutations:
		which_mut = m.split('(')[0].strip()
		if which_mut in known_mutations:
			effect = m.split('(')[1].split(')')[0].strip()
			if effect != known_mutations[which_mut]:
				warnings.WarningMessage('mismatched mutation effect', which_mut, effect)
			if 'poor' in effect:
				out.append(3)
			elif 'good' in effect:
				out.append(2)
			else:
				out.append(1)
	return out


def load_mutations(flnm):
	out = {}
	with open(flnm, 'r') as F:
		for r in F.readlines():
			if not r.startswith('#'):
				if '->' in r:
					mutation = r.split('->')[0].strip()
					effect = r.split('\n')[0].split('->')[1].strip()
					out[mutation] = effect
	return out


def interpret_value(var, vl):
	if var == 'Recurrence' or var == 'Known drug resistance' or var == 'Interval debulking':
		if vl == 'no':
			return 0
		else:
			if '(' in vl:
				specification = vl.split('(')[1].split(')')[0]
				if specification == 'not known':
					return 1
				else:
					if var == 'Recurrence':
						specification = int(specification)
					return [1, specification]
	elif var == 'Age' or var == 'CA125':
		if '>' in vl:
			vl = vl.split('>')[1]
		return float(vl)
	elif var == 'Stage':
		return get_stage(vl)
	else:
		if vl == 'no':
			return 0
		else:
			return vl


def get_stage(st):
	stages = {'I': 1, 'IA': 1.1, 'IB': 1.2, 'IC': 1.3, 'II': 2, 'IIA': 2.1, 'IIB': 2.2, 'IIC': 2.3,
			  'III': 3, 'IIIA': 3.1, 'IIIB': 3.2, 'IIIC': 3.3, 'IV': 4, 'IVA': 4.1, 'IVB': 4.2, 'IVC': 4.3}
	return stages[st]
