import ALISON.simulator
import ALISON.utility

cell_file_config = 'fibroblast_cells_config_opt.txt'
configuration_file = 'fibroblasts_configuration.txt'
cell_file_simul = 'fibroblasts_cells_config.txt'
parameters = [0.0001, 0.001, 0.01, 0.1, 1]
replicates = 3
configuration = ALISON.utility.get_configuration(cell_file_config, parameters)
ALISON.utility.print_file(cell_file_config, configuration, cell_file_simul)
for r in range(replicates):
	structure = ALISON.simulator.ALISON(configuration_file)
	name = 'replicate_' + str(r) + '_parameters_'
	for c in configuration:
		name = name + c + '_'+ str(configuration[c])+'_'
	structure.simulate(name[:-1])
while configuration != -1:
	configuration = ALISON.utility.get_configuration(cell_file_config,parameters, current_set= configuration)
	ALISON.utility.print_file(cell_file_config, configuration, cell_file_simul)
	for r in range(replicates):
		structure = ALISON.simulator.ALISON(configuration_file)
		name = 'replicate_' + str(r) + '_parameters_'
		for c in configuration:
			name = name + c + '_'+ str(configuration[c])+'_'
		structure.simulate(name[:-1])

