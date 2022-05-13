import os
import warnings
import numpy as np
import pyvista as pv
from scipy.sparse import lil_array, csr_array


class Mesh:
	def __init__(self, file_name):
		self.folder = os.getcwd() + os.path.sep + 'meshes' + os.path.sep + file_name + os.path.sep
		nodes = self.read_nodes(self.folder + file_name + '.node')
		elements = self.read_elements(self.folder + file_name + '.ele')
		self.material_properties = self.read_properties(self.folder + file_name + '.matprop')
		self.mesh = pv.PolyData(nodes, elements)
		self.nodes_in_elements = self.get_element_nodes(self.mesh)
		self.parameters_elements = self.get_parameters_elements(self.mesh, self.nodes_in_elements)
		self.jacobian = self.get_jacobian(self.mesh, self.nodes_in_elements)
		self.ns = self.compute_ns(self.parameters_elements, self.nodes_in_elements)
		self.m = csr_array((self.mesh.n_points, self.mesh.n_points))
		self.k = csr_array((self.mesh.n_points, self.mesh.n_points))
		for e in range(self.mesh.n_faces):
			print(e)
			me = self.get_me(self.mesh, self.material_properties, e, self.nodes_in_elements[e], self.jacobian[e], self.ns)
			ke = self.get_ke(self.mesh, self.material_properties, self.nodes_in_elements[e], self.jacobian[e])
			self.m = np.add(self.m, me.tocsr())
			self.k = np.add(self.k, ke.tocsr())

	@staticmethod
	def compute_ns(parameters, nodes_elements):
		integration_points = [[0.1381966, 0.1381966, 0.1381966], [0.58541020, 0.1381966, 0.1381966],
							  [0.1381966, 0.58541020, 0.1381966], [0.1381966, 0.1381966, 0.58541020]]
		out = {}
		for el in nodes_elements:
			pars = parameters[el]
			out[el] = {}
			for idx in range(len(nodes_elements[el])):
				out[el][idx] = []
				for ip in integration_points:
					if idx > 0:
						[a, b, c, d] = pars[idx]
						out[el][idx].append(a + b * ip[0] + c * ip[1] + d * ip[2])
					else:
						n = 1
						for ii in pars:
							if ii != idx:
								[a, b, c, d] = pars[ii]
								n -= a + b * ip[0] + c * ip[1] + d * ip[2]
						out[el][idx].append(n)
		return out

	@staticmethod
	def get_element_nodes(mesh):
		# Function that extracts the nodes belonging to each element
		out = {}
		nodes_per_element = mesh.faces[0]
		for idx in range(mesh.n_faces):
			idx_start = idx * (nodes_per_element + 1)
			out[idx] = mesh.faces[idx_start + 1:idx_start + 1 + nodes_per_element]
		return out

	@staticmethod
	def get_jacobian(mesh, nodes_el):
		out = {}
		for e in nodes_el:
			out[e] = np.zeros((3, 3))
			for v in range(3):
				cs = mesh.points[nodes_el[e], v]
				out[e][:, v] = cs[1:] - cs[0]
		return out

	@staticmethod
	def get_ke(mesh, material, nodes_el, j):
		out_k = lil_array((mesh.n_points, mesh.n_points))
		inv_j = np.linalg.inv(j)

		w = 0.0416667
		dn = [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

		for ii, i in enumerate(nodes_el):
			for jj, j in enumerate(nodes_el):
				dni_0 = sum(dn[ii] * inv_j[0])
				dni_1 = sum(dn[ii] * inv_j[1])
				dni_2 = sum(dn[ii] * inv_j[2])
				dnj_0 = sum(dn[jj] * inv_j[0])
				dnj_1 = sum(dn[jj] * inv_j[1])
				dnj_2 = sum(dn[jj] * inv_j[2])
				dnm = (dni_0 + dni_1 + dni_2) * (dnj_0 + dnj_1 + dnj_2)
				out_k[i, j] = 4 * w * material['k'] * dnm
		return out_k

	@staticmethod
	def get_me(mesh, material, el, nodes_el, j, ns):
		out_m = lil_array((mesh.n_points, mesh.n_points))
		det_j = np.linalg.det(j)
		rhocv = material['rho'] * material['cv']
		integration_points = [[0.1381966, 0.1381966, 0.1381966], [0.58541020, 0.1381966, 0.1381966],
							  [0.1381966, 0.58541020, 0.1381966], [0.1381966, 0.1381966, 0.58541020]]
		w = 0.0416667
		for jj, j in enumerate(nodes_el):
			for ii, i in enumerate(nodes_el):
				temp = []
				for ip in range(len(integration_points)):
					nj = ns[el][jj][ip]
					ni = ns[el][ii][ip]
					temp.append(rhocv * ni * nj * det_j * w)  # do I need det_j? is it correct?
				out_m[i, j] = sum(temp)
		return out_m

	@staticmethod
	def get_parameters_elements(mesh, element_nodes):
		# Function that computes the parameters of the shape functions approximating the field within each element of the mesh.
		# As noted in the print this formulation assumes the mesh to be tetrahedral and 3D
		warnings.warn(
			'This formulation of get_parameters is only appropriate if you have a 3D tetrahedral mesh If that '
			'is not the case, you must modify this function')
		out = {}
		row = {0: [1, 2, 3], 1: [2, 3, 0], 2: [3, 0, 1],
			   3: [0, 1, 2]}  # As detailed in notebook 2 and the Zieukievicz book,
		# a, b, c, d can be obtained as ratios of determinants and these are the indices of row and column of the submatrices
		col = {'a': [1, 2, 3], 'b': [0, 2, 3], 'c': [1, 0, 3], 'd': [1, 2, 0]}

		for nn in range(mesh.n_faces):
			out[nn] = {}
			nodes_e = element_nodes[nn]
			matrix = np.ones((len(nodes_e), len(nodes_e)))
			for ii, i in enumerate(nodes_e):
				matrix[ii, 1:] = mesh.points[i, :]
			sixv = np.linalg.det(matrix)
			for r in range(len(row)):
				a = np.linalg.det(Mesh.get_submatrix(matrix, row[r], col['a']))
				b = -np.linalg.det(Mesh.get_submatrix(matrix, row[r], col['b']))
				c = -np.linalg.det(Mesh.get_submatrix(matrix, row[r], col['c']))
				d = -np.linalg.det(Mesh.get_submatrix(matrix, row[r], col['d']))
				out[nn][r] = [a / sixv, b / sixv, c / sixv, d / sixv]
		return out

	@staticmethod
	def get_submatrix(mtrx, r_id, c_id):
		# Function that isolates the submatrix corresponding to the rows and columns indices in r_id and c_id
		out = np.zeros((len(r_id), len(c_id)))
		for ir, r in enumerate(r_id):
			for ic, c in enumerate(c_id):
				out[ir, ic] = mtrx[r, c]
		return out

	@staticmethod
	def read_elements(filepath):
		# Function that reads the element file of the mesh.
		with open(filepath) as F:
			for ir, r in enumerate(F.readlines()):
				if ir == 0:
					length = int(r.split('\n')[0].split()[0])
					nele = int(r.split('\n')[0].split()[1])
					out = np.zeros((length, nele + 1))
				else:
					tmp = r.split('\n')[0].split()
					if len(tmp) == 5:
						tmp = [int(x) for x in tmp]
						out[tmp[0], :] = [nele] + tmp[1:]
			return np.hstack(out).astype(int)

	@staticmethod
	def read_nodes(file_path):
		# function that reads the nodes file of the mesh.
		with open(file_path) as F:
			for ir, r in enumerate(F.readlines()):
				if '#' not in r:
					temp = [float(x) for x in r.split('\n')[0].split()]
				if ir == 0:
					out = np.zeros((int(temp[0]), int(temp[1])))
				else:
					out[int(temp[0]), :] = temp[1:]
			return out

	@staticmethod
	def read_properties(file_path):
		out = {}
		with open(file_path) as F:
			for r in F.readlines():
				if '#' not in r:
					temp = r.split('\n')[0].split('->')
					out[temp[0].strip()] = float(temp[1])
		return out
