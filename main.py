from ALISON import simulator as organoid
import dill

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

	#configuration_file = 'organoid_configuration.txt'
	#structure = organoid.ALISON(configuration_file)
	#fileOut = 'debugging_structure.pickle'
	#with open(fileOut, 'wb') as F:
	#	dill.dump(structure, F)
	file_name = 'debugging_structure.pickle'
	with open(file_name, 'rb') as F:
		structure = dill.load(F)
	structure.simulate()
