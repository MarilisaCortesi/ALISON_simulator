import os


class ALISON_clonal:
	def __init__(self, configuration_file):
		self.base_name = configuration_file.split('.txt')[0]
		full_path_configuration = os.getcwd() + os.path.sep + 'experiment_configuration_files' + os.path.sep \
								  + self.base_name + '.txt'