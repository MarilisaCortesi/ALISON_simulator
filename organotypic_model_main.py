import ALISON.simulator
import ALISON.utility


configuration_file = 'organoid_configuration.txt'
structure = ALISON.simulator.ALISON(configuration_file)
name = 'replicate_'+str(0)+ '_configuration_'+ configuration_file.split('.txt')[0]
_ = structure.simulate(name)
