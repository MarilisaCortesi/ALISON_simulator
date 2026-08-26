import pickle

from Bio import Phylo
import os

'''
test_tree = '/Users/marilisacortesi/Desktop/clonal_evolution/newick/newick_genetic_tree_starting_concentration_10_replicate_2.dnd'

tree = Phylo.read(test_tree, 'newick')
Phylo.draw(tree, branch_labels=lambda c: c.branch_length)
'''

folder = '/Users/marilisacortesi/Desktop/clonal_evolution/newick/'
files = os.listdir(folder)
data = {}
for f in files:
	if f.startswith('.'):
		continue
	starting_conc = int(f.split('concentration_')[1].split('_')[0])
	replicate = int(f.split('replicate_')[1].split('.dnd')[0])
	if starting_conc not in data:
		data[starting_conc] = {}
	with open(folder+f, 'r') as F:
		for r in F.readlines():
			data[starting_conc][replicate] = r

print('s')