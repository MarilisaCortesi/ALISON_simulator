import dill
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def prepare_animation(bar_container):
	def animate(frame_number):
		print(frame_number)
		data3 = data2[frame_number]['mesh'][variable]
		n, _ = np.histogram(data3, hist_bins)
		for count, rect in zip(n, bar_container.patches):
			rect.set_height(count)
		return bar_container.patches
	return animate


def convert_data(data, K):
	out = {}
	for kk, k in enumerate(sorted(K)):
		out[kk] = data[k]
	return out

def get_cells(data):
	out = {}
	K = list(data.keys())
	n_its = len(K)
	init_c = data['initial_condition']['cell_population']
	for c in init_c:
		if c.type not in out:
			out[c.type] = {}
			out[c.type][c.status] = np.zeros(n_its)
			out[c.type][c.status][0] += 1
		else:
			if c.status not in out[c.type]:
				out[c.type][c.status] = np.zeros(n_its)
				out[c.type][c.status][0] += 1
			else:
				out[c.type][c.status][0] += 1
	K.remove('initial_condition')
	for kk, k in enumerate(sorted(K)):
		cp = data[k]['cell_population']
		for c in cp:
			if c.type not in out:
				out[c.type] = {}
				out[c.type][c.status] = np.zeros(n_its)
				out[c.type][c.status][kk+1] += 1
			else:
				if c.status not in out[c.type]:
					out[c.type][c.status] = np.zeros(n_its)
					out[c.type][c.status][kk + 1] += 1
				else:
					out[c.type][c.status][kk + 1] += 1
	return out

def plot_cells(cells):
	colors = {1: 'r', 2: 'b', 3: 'g'}
	for c in cells:
		f, ax = plt.subplots()
		ax.set_title(c)
		for cc in cells[c]:
			ax.plot(range(len(cells[c][cc])), cells[c][cc], marker='o', color=colors[cc])



filename = 'outputs/14102022_12:27:58_complete_simulation.pickle'
variable = 'oxygen'
with open(filename, 'rb') as F:
	data = dill.load(F)
K = list(data.keys())
K.remove('initial_condition')
data2 = convert_data(data, K)
if variable == 'glucose':
	hist_bins = np.linspace(5e-7, 1e-6, 100)
elif variable == 'lactate':
	hist_bins = np.linspace(0, 1e-7, 100)
elif variable == 'oxygen':
	hist_bins = np.linspace(0, 1e-3, 100)
else:
	raise ValueError('unrecognised variable')

f, ax = plt.subplots()
_, _, bar_container = ax.hist(data['initial_condition']['mesh'][variable], hist_bins, lw=1, ec='b', fc='r', alpha=0.5)
ax.set_ylim(top=1e5)
ani = animation.FuncAnimation(f, prepare_animation(bar_container), len(K), repeat=False, blit=True)

cells = get_cells(data)
plot_cells(cells)
plt.show()
