import ALISON.simulator
import ALISON.utility


configuration_file = 'organoid_configuration.txt'
structure = ALISON.simulator.ALISON(configuration_file)
structure.simulate()
