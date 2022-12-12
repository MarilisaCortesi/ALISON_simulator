import os
import warnings
import itertools
import sympy
import numpy as np
import pyvista as pv
import scipy.sparse.linalg
from scipy.sparse import lil_array, csr_array
import ALISON.utility as utility


class Mesh:
	def __init__(self, file_name):
		warnings.warn(
			'Thank you for using the mesh configurator. This will take a while, time to take a break or work on another task')
		self.folder = os.getcwd() + os.path.sep + 'meshes' + os.path.sep + file_name + os.path.sep
		nodes, feature = self.read_nodes(self.folder + file_name + '.node')
		nodes = self.scale_cylinder(nodes, 3.5, 5.3)  # TODO add them as inputs
		elements = self.read_elements(self.folder + file_name + '.ele')
		# elements = self.sort_nodes(nodes, elements)
		self.material_properties = self.read_properties(self.folder + file_name + '.matprop')
		self.mesh = pv.PolyData(nodes, elements)
		print('mesh')
		if len(feature) > 0:
			self.mesh.field_data['cells'] = feature
		self.nodes_in_elements = self.get_element_nodes(self.mesh)
		self.neighbours = self.get_neighbours(self.mesh, self.nodes_in_elements)
		print('neighbours')
		# self.parameters_elements, self.volume_element = self.get_parameters_elements(self.mesh, self.nodes_in_elements)
		# self.jacobian = self.get_jacobian(self.mesh, self.nodes_in_elements)
		self.alphas = self.get_alphas(self.mesh, self.nodes_in_elements)
		print('alphas')
		'''
		self.ns = self.compute_ns(self.parameters_elements, self.nodes_in_elements)
		self.m = csr_array((self.mesh.n_points, self.mesh.n_points))
		self.k = csr_array((self.mesh.n_points, self.mesh.n_points))
		
		for e in range(self.mesh.n_faces):
			print(e)
			me = self.get_me(self.mesh, self.material_properties, e, self.nodes_in_elements[e], self.jacobian[e],
							 self.ns, self.volume_element)
			ke = self.get_ke(self.mesh, self.material_properties, self.nodes_in_elements[e], self.jacobian[e], self.volume_element)
			self.m = np.add(self.m, me.tocsr())
			self.k = np.add(self.k, ke.tocsr())
		'''
		self.ke_symbol = self.get_ke_symbol(self.alphas, self.material_properties['k'])
		self.k_symbol = self.combine_matrix(self.ke_symbol, self.nodes_in_elements, self.mesh.n_points)
		print('k')
		#self.ke_pyeit = self.get_ke_pyeit(self.mesh, self.nodes_in_elements)
		#self.k_pyeit = self.get_k_pyeit(self.ke_pyeit, self.nodes_in_elements, self.material_properties['k'],
		#								self.mesh.n_points)
		self.me_symbol = self.get_me_symbol(self.alphas,
											self.material_properties['rho'] * self.material_properties['cv'])
		self.m_symbol = self.combine_matrix(self.me_symbol, self.nodes_in_elements, self.mesh.n_points)
		#self.inv_m05k = scipy.sparse.linalg.inv(self.m_symbol + 0.5 * self.k_symbol)



	@staticmethod
	def get_alphas(msh, nds_els):
		out = {}
		a1, a2, a3, a4 = sympy.symbols('a1, a2, a3, a4')
		t1, t2, t3, t4 = sympy.symbols('t1, t2, t3, t4')
		for e in range(msh.n_faces):
			nds = nds_els[e]
			x1, y1, z1 = msh.points[nds[0], :]
			x2, y2, z2 = msh.points[nds[1], :]
			x3, y3, z3 = msh.points[nds[2], :]
			x4, y4, z4 = msh.points[nds[3], :]
			eq1 = sympy.Eq(a1 + a2 * x1 + a3 * y1 + a4 * z1, t1)
			eq2 = sympy.Eq(a1 + a2 * x2 + a3 * y2 + a4 * z2, t2)
			eq3 = sympy.Eq(a1 + a2 * x3 + a3 * y3 + a4 * z3, t3)
			eq4 = sympy.Eq(a1 + a2 * x4 + a3 * y4 + a4 * z4, t4)
			solution = sympy.solve((eq1, eq2, eq3, eq4), (a1, a2, a3, a4))
			out[e] = solution
		return out

	@staticmethod
	def combine_matrix(xes, nde, nds):
		x = lil_array((nds, nds))
		for n in nde:
			nodes_element = nde[n]
			ke_element = xes[n]
			for ii, i in enumerate(nodes_element):
				for jj, j in enumerate(nodes_element):
					x[i, j] += ke_element[ii, jj]
		return x

	@staticmethod
	def get_ke_symbol(alpha_values, k_value):
		a2, a3, a4 = sympy.symbols('a2, a3, a4')
		t1, t2, t3, t4 = sympy.symbols('t1, t2, t3, t4')
		out = {}
		for a in alpha_values:
			ke = np.zeros((len(alpha_values[a]), len(alpha_values[a])))
			bs = alpha_values[a][a2].as_coefficients_dict()
			cs = alpha_values[a][a3].as_coefficients_dict()
			ds = alpha_values[a][a4].as_coefficients_dict()

			ke[0, 0] = k_value * (bs[t1] ** 2 + cs[t1] ** 2 + ds[t1] ** 2)
			ke[1, 1] = k_value * (bs[t2] ** 2 + cs[t2] ** 2 + ds[t2] ** 2)
			ke[2, 2] = k_value * (bs[t3] ** 2 + cs[t3] ** 2 + ds[t3] ** 2)
			ke[3, 3] = k_value * (bs[t4] ** 2 + cs[t4] ** 2 + ds[t4] ** 2)

			ke[0, 1] = k_value * (bs[t1] * bs[t2] + cs[t1] * cs[t2] + ds[t1] * ds[t2])
			ke[1, 0] = ke[0, 1]

			ke[0, 2] = k_value * (bs[t1] * bs[t3] + cs[t1] * cs[t3] + ds[t1] * ds[t3])
			ke[2, 0] = ke[0, 2]

			ke[0, 3] = k_value * (bs[t1] * bs[t4] + cs[t1] * cs[t4] + ds[t1] * ds[t4])
			ke[3, 0] = ke[0, 3]

			ke[1, 2] = k_value * (bs[t2] * bs[t3] + cs[t2] * cs[t3] + ds[t2] * ds[t3])
			ke[2, 1] = ke[1, 2]

			ke[1, 3] = k_value * (bs[t2] * bs[t4] + cs[t2] * cs[t4] + ds[t2] * ds[t4])
			ke[3, 1] = ke[1, 3]

			ke[2, 3] = k_value * (bs[t3] * bs[t4] + cs[t3] * cs[t4] + ds[t3] * ds[t4])
			ke[3, 2] = ke[2, 3]
			out[a] = ke
		return out

	@staticmethod
	def get_me_symbol(alphas, rhocv):
		a1, a2, a3, a4 = sympy.symbols('a1, a2, a3, a4')
		t1, t2, t3, t4 = sympy.symbols('t1, t2, t3, t4')
		x, y, z = sympy.symbols('x, y, z')
		out = {}
		for a in alphas:
			me = np.zeros((len(alphas[a]), len(alphas[a])))
			a_s = alphas[a][a1].as_coefficients_dict()
			bs = alphas[a][a2].as_coefficients_dict()
			cs = alphas[a][a3].as_coefficients_dict()
			ds = alphas[a][a4].as_coefficients_dict()

			me00 = rhocv * (
					bs[t1] ** 2 * x ** 2 + cs[t1] ** 2 * y ** 2 + ds[t1] ** 2 * z ** 2 + 2 * a_s[t1] * bs[t1] * x
					+ 2 * a_s[t1] * cs[t1] * y + 2 * a_s[t1] * ds[t1] * z + 2 * bs[t1] * cs[t1] * x * y
					+ 2 * bs[t1] * ds[t1] * x * z + 2 * cs[t1] * ds[t1] * y * z + a_s[t1] ** 2)
			me[0, 0] = sympy.integrate(me00, (x, 0, 1), (y, 0, 1), (z, 0, 1))

			me11 = rhocv * (
					bs[t2] ** 2 * x ** 2 + cs[t2] ** 2 * y ** 2 + ds[t2] ** 2 * z ** 2 + 2 * a_s[t2] * bs[t2] * x
					+ 2 * a_s[t2] * cs[t2] * y + 2 * a_s[t2] * ds[t2] * z + 2 * bs[t2] * cs[t2] * x * y
					+ 2 * bs[t2] * ds[t2] * x * z + 2 * cs[t2] * ds[t2] * y * z + a_s[t2] ** 2)
			me[1, 1] = sympy.integrate(me11, (x, 0, 1), (y, 0, 1), (z, 0, 1))

			me22 = rhocv * (
					bs[t3] ** 2 * x ** 2 + cs[t3] ** 2 * y ** 2 + ds[t3] ** 2 * z ** 2 + 2 * a_s[t3] * bs[t3] * x
					+ 2 * a_s[t3] * cs[t3] * y + 2 * a_s[t3] * ds[t3] * z + 2 * bs[t3] * cs[t3] * x * y
					+ 2 * bs[t3] * ds[t3] * x * z + 2 * cs[t3] * ds[t3] * y * z + a_s[t3] ** 2)
			me[2, 2] = sympy.integrate(me22, (x, 0, 1), (y, 0, 1), (z, 0, 1))

			me33 = rhocv * (
					bs[t1] ** 2 * x ** 2 + cs[t4] ** 2 * y ** 2 + ds[t4] ** 2 * z ** 2 + 2 * a_s[t4] * bs[t4] * x
					+ 2 * a_s[t4] * cs[t4] * y + 2 * a_s[t4] * ds[t4] * z + 2 * bs[t4] * cs[t4] * x * y
					+ 2 * bs[t4] * ds[t4] * x * z + 2 * cs[t4] * ds[t4] * y * z + a_s[t4] ** 2)
			me[3, 3] = sympy.integrate(me33, (x, 0, 1), (y, 0, 1), (z, 0, 1))

			me01 = rhocv * (
					bs[t1] * bs[t2] * x ** 2 + cs[t1] * cs[t2] * y ** 2 + ds[t1] * ds[t2] * z ** 2 +
					(a_s[t1] * bs[t2] + a_s[t2] * bs[t1]) * x + (a_s[t1] * cs[t2] + a_s[t2] * cs[t1]) * y
					+ (a_s[t1] * ds[t2] + a_s[t2] * ds[t1]) * z + (bs[t1] * cs[t2] + bs[t2] * cs[t1]) * x * y
					+ (bs[t1] * ds[t2] + bs[t2] * ds[t1]) * x * z + (cs[t1] * ds[t2] + cs[t2] * ds[t1]) * y * z
					+ a_s[t1] * a_s[t2])

			me[0, 1] = sympy.integrate(me01, (x, 0, 1), (y, 0, 1), (z, 0, 1))
			me[1, 0] = me[0, 1]

			me02 = rhocv * (
					bs[t1] * bs[t3] * x ** 2 + cs[t1] * cs[t3] * y ** 2 + ds[t1] * ds[t3] * z ** 2 +
					(a_s[t1] * bs[t3] + a_s[t3] * bs[t1]) * x + (a_s[t1] * cs[t3] + a_s[t3] * cs[t1]) * y
					+ (a_s[t1] * ds[t3] + a_s[t3] * ds[t1]) * z + (bs[t1] * cs[t3] + bs[t3] * cs[t1]) * x * y
					+ (bs[t1] * ds[t3] + bs[t3] * ds[t1]) * x * z + (cs[t1] * ds[t3] + cs[t3] * ds[t1]) * y * z
					+ a_s[t1] * a_s[t3])

			me[0, 2] = sympy.integrate(me02, (x, 0, 1), (y, 0, 1), (z, 0, 1))
			me[2, 0] = me[0, 2]

			me03 = rhocv * (
					bs[t1] * bs[t4] * x ** 2 + cs[t1] * cs[t4] * y ** 2 + ds[t1] * ds[t4] * z ** 2 +
					(a_s[t1] * bs[t4] + a_s[t4] * bs[t1]) * x + (a_s[t1] * cs[t4] + a_s[t4] * cs[t1]) * y
					+ (a_s[t1] * ds[t4] + a_s[t4] * ds[t1]) * z + (bs[t1] * cs[t4] + bs[t4] * cs[t1]) * x * y
					+ (bs[t1] * ds[t4] + bs[t4] * ds[t1]) * x * z + (cs[t1] * ds[t4] + cs[t4] * ds[t1]) * y * z
					+ a_s[t1] * a_s[t4])

			me[0, 3] = sympy.integrate(me03, (x, 0, 1), (y, 0, 1), (z, 0, 1))
			me[3, 0] = me[0, 3]

			me12 = rhocv * (
					bs[t2] * bs[t3] * x ** 2 + cs[t2] * cs[t3] * y ** 2 + ds[t2] * ds[t3] * z ** 2 +
					(a_s[t2] * bs[t3] + a_s[t3] * bs[t2]) * x + (a_s[t2] * cs[t3] + a_s[t3] * cs[t2]) * y
					+ (a_s[t2] * ds[t3] + a_s[t3] * ds[t2]) * z + (bs[t2] * cs[t3] + bs[t3] * cs[t2]) * x * y
					+ (bs[t2] * ds[t3] + bs[t3] * ds[t2]) * x * z + (cs[t2] * ds[t3] + cs[t3] * ds[t2]) * y * z
					+ a_s[t2] * a_s[t3])

			me[1, 2] = sympy.integrate(me12, (x, 0, 1), (y, 0, 1), (z, 0, 1))
			me[2, 1] = me[1, 2]

			me13 = rhocv * (
					bs[t2] * bs[t4] * x ** 2 + cs[t2] * cs[t4] * y ** 2 + ds[t2] * ds[t4] * z ** 2 +
					(a_s[t2] * bs[t4] + a_s[t4] * bs[t2]) * x + (a_s[t2] * cs[t4] + a_s[t4] * cs[t2]) * y
					+ (a_s[t2] * ds[t4] + a_s[t4] * ds[t2]) * z + (bs[t2] * cs[t4] + bs[t4] * cs[t2]) * x * y
					+ (bs[t2] * ds[t4] + bs[t4] * ds[t2]) * x * z + (cs[t2] * ds[t4] + cs[t4] * ds[t2]) * y * z
					+ a_s[t2] * a_s[t4])

			me[1, 3] = sympy.integrate(me13, (x, 0, 1), (y, 0, 1), (z, 0, 1))
			me[3, 1] = me[1, 3]

			me23 = rhocv * (
					bs[t3] * bs[t4] * x ** 2 + cs[t3] * cs[t4] * y ** 2 + ds[t3] * ds[t4] * z ** 2 +
					(a_s[t3] * bs[t4] + a_s[t4] * bs[t3]) * x + (a_s[t3] * cs[t4] + a_s[t4] * cs[t3]) * y
					+ (a_s[t3] * ds[t4] + a_s[t4] * ds[t3]) * z + (bs[t3] * cs[t4] + bs[t4] * cs[t3]) * x * y
					+ (bs[t3] * ds[t4] + bs[t4] * ds[t3]) * x * z + (cs[t3] * ds[t4] + cs[t4] * ds[t3]) * y * z
					+ a_s[t3] * a_s[t4])

			me[2, 3] = sympy.integrate(me23, (x, 0, 1), (y, 0, 1), (z, 0, 1))
			me[3, 2] = me[2, 3]

			out[a] = me
		return out

	@staticmethod
	def get_m_pyeit(me, elem_dict, mat_prop, npts):
		elem = np.zeros((len(elem_dict), 4))
		for n in elem_dict:
			elem[n, :] = [int(x) for x in elem_dict[n]]
		n_tri, n_vertices = elem.shape

		# New: use IJV indexed sparse matrix to assemble K (fast, prefer)
		# index = np.array([np.meshgrid(no, no, indexing='ij') for no in tri])
		# note: meshgrid is slow, using handcraft sparse index, for example
		# let tri=[[1, 2, 3], [4, 5, 6]], then indexing='ij' is equivalent to
		# row = [1, 1, 1, 2, 2, 2, ...]
		# col = [1, 2, 3, 1, 2, 3, ...]
		row = np.repeat(elem, n_vertices).ravel()
		col = np.repeat(elem, n_vertices, axis=0).ravel()
		rho_cv = mat_prop['rho'] * mat_prop['cv']
		data = np.array([me[i] * rho_cv for i in range(n_tri)]).ravel()
		return scipy.sparse.csr_matrix((data, (row, col)), shape=(npts, npts))

	@staticmethod
	def get_k_pyeit(ke, elem_dict, k_val, n_pts):
		elem = np.zeros((len(elem_dict), 4))
		for n in elem_dict:
			elem[n, :] = [int(x) for x in elem_dict[n]]
		n_tri, n_vertices = elem.shape

		# New: use IJV indexed sparse matrix to assemble K (fast, prefer)
		# index = np.array([np.meshgrid(no, no, indexing='ij') for no in tri])
		# note: meshgrid is slow, using handcraft sparse index, for example
		# let tri=[[1, 2, 3], [4, 5, 6]], then indexing='ij' is equivalent to
		# row = [1, 1, 1, 2, 2, 2, ...]
		# col = [1, 2, 3, 1, 2, 3, ...]
		row = np.repeat(elem, n_vertices).ravel()
		col = np.repeat(elem, n_vertices, axis=0).ravel()
		data = np.array([ke[i] * k_val for i in range(n_tri)]).ravel()

		# for efficient sparse inverse (csc)
		return scipy.sparse.csr_matrix((data, (row, col)), shape=(n_pts, n_pts))

	@staticmethod
	def get_me_pyeit(mesh, nds):
		tri = np.zeros((mesh.n_faces, 4))
		for n in nds:
			tri[n, :] = [int(x) for x in nds[n]]
		nodes = mesh.points
		n_tri, n_vertices = tri.shape

		# default data types for ke
		me_array = np.zeros((n_tri, n_vertices, n_vertices))
		for ei in range(n_tri):
			no = [int(x) for x in tri[ei, :]]
			xy = nodes[no]

			# compute the KIJ (permittivity=1.)
			me = Mesh.m_tetrahedron(xy)
			me_array[ei] = me

		return me_array

	@staticmethod
	def get_ke_pyeit(mesh, nds):
		tri = np.zeros((mesh.n_faces, 4))
		for n in nds:
			tri[n, :] = [int(x) for x in nds[n]]
		nodes = mesh.points
		n_tri, n_vertices = tri.shape

		# default data types for ke
		ke_array = np.zeros((n_tri, n_vertices, n_vertices))
		for ei in range(n_tri):
			no = [int(x) for x in tri[ei, :]]
			xy = nodes[no]

			# compute the KIJ (permittivity=1.)
			ke = Mesh.k_tetrahedron(xy)
			ke_array[ei] = ke

		return ke_array

	@staticmethod
	def sort_nodes(nodes, elements):
		out = np.zeros_like(elements)
		nodes_per_element = elements[0]
		total_elements = len(elements) / (elements[0] + 1)
		out_idx = 0
		for idx in range(int(total_elements)):
			idx_start = idx * (nodes_per_element + 1)
			nodes_in_element = elements[idx_start + 1:idx_start + 1 + nodes_per_element]
			idx_1 = [2, 3, 0, 1]
			idx_2 = [1, 2, 3, 0]
			s = np.zeros((4, 3))
			for i in range(4):
				s[i] = nodes[nodes_in_element[idx_1[i]]] - nodes[nodes_in_element[idx_2[i]]]
			vt = 1.0 / 6 * np.linalg.det(s[[0, 1, 2]])
			if vt < 0:
				nodes_in_element = [nodes_in_element[1], nodes_in_element[2], nodes_in_element[3], nodes_in_element[0]]
				s = np.zeros((4, 3))
				for i in range(4):
					s[i] = nodes[nodes_in_element[idx_1[i]]] - nodes[nodes_in_element[idx_2[i]]]
				vt = 1.0 / 6 * np.linalg.det(s[[0, 1, 2]])
				if vt < 0:
					raise ValueError('the barycentric coordinate is negative')
				else:
					out[out_idx] = 4
					out[out_idx + 1: out_idx + 5] = nodes_in_element
					out_idx += 5

			else:
				out[out_idx] = 4
				out[out_idx + 1: out_idx + 5] = nodes_in_element
				out_idx += 5
		return out

	@staticmethod
	def m_tetrahedron(xy: np.ndarray) -> np.ndarray:
		"""
		Given a point-matrix of an element, solving for Kij analytically
		using barycentric coordinates (simplex coordinates)
		Parameters
		----------
		xy: np.ndarray
			(x,y) of nodes 1, 2, 3, 4 given in counterclockwise manner, see notes.
		Returns
		-------
		np.ndarray
			local stiffness matrix
		Notes
		-----
		A tetrahedron is described using [0, 1, 2, 3] (local node index) or
		[171, 27, 9, 53] (global index). Counterclockwise (CCW) is defined
		such that the barycentric coordinate of face (1->2->3) is positive.
		"""
		s = xy[[2, 3, 0, 1]] - xy[[1, 2, 3, 0]]

		# volume of the tetrahedron, Note abs is removed since version 2020,
		# user must make sure all tetrahedrons are CCW (counter clock wised).
		vt = np.absolute(1.0 / 6 * np.linalg.det(s[[1, 2, 3]]))
		if vt < 0:
			xy = np.array([xy[1], xy[2], xy[3], xy[0]])
			s = xy[[2, 3, 0, 1]] - xy[[1, 2, 3, 0]]
			vt = 1.0 / 6 * np.linalg.det(s[[0, 1, 2]])
			if vt < 0:
				raise ValueError('the barycentric coordinate is negative')

		# calculate area (vector) of triangle faces
		# re-normalize using alternative (+,-) signs
		ij_pairs = [[0, 1], [1, 2], [2, 3], [3, 0]]
		signs = [1, -1, 1, -1]
		a = np.array([sign * np.cross(s[i], s[j]) for (i, j), sign in zip(ij_pairs, signs)])

		# local (e for element) stiffness matrix
		return np.dot(a, a.transpose()) / (36.0 * vt)

	@staticmethod
	def k_tetrahedron(xy: np.ndarray) -> np.ndarray:
		"""
		Given a point-matrix of an element, solving for Kij analytically
		using barycentric coordinates (simplex coordinates)
		Parameters
		----------
		xy: np.ndarray
			(x,y) of nodes 1, 2, 3, 4 given in counterclockwise manner, see notes.
		Returns
		-------
		np.ndarray
			local stiffness matrix
		Notes
		-----
		A tetrahedron is described using [0, 1, 2, 3] (local node index) or
		[171, 27, 9, 53] (global index). Counterclockwise (CCW) is defined
		such that the barycentric coordinate of face (1->2->3) is positive.
		"""
		s = xy[[2, 3, 0, 1]] - xy[[1, 2, 3, 0]]

		# volume of the tetrahedron, Note abs is removed since version 2020,
		# user must make sure all tetrahedrons are CCW (counter clock wised).
		vt = np.absolute(1.0 / 6 * np.linalg.det(s[[1, 2, 3]]))
		if vt < 0:
			xy = np.array([xy[1], xy[2], xy[3], xy[0]])
			s = xy[[2, 3, 0, 1]] - xy[[1, 2, 3, 0]]
			vt = 1.0 / 6 * np.linalg.det(s[[0, 1, 2]])
			if vt < 0:
				raise ValueError('the barycentric coordinate is negative')

		# calculate area (vector) of triangle faces
		# re-normalize using alternative (+,-) signs
		ij_pairs = [[0, 1], [1, 2], [2, 3], [3, 0]]
		signs = [1, -1, 1, -1]
		a = np.array([sign * np.cross(s[i], s[j]) for (i, j), sign in zip(ij_pairs, signs)])

		# local (e for element) stiffness matrix
		return np.dot(a, a.transpose()) / (36.0 * vt)

	@staticmethod
	def compute_ns(parameters, nodes_elements):
		integration_points = [[0.1381966011250105, 0.1381966011250105, 0.1381966011250105],
							  [0.5854101966249685, 0.1381966011250105, 0.1381966011250105],
							  [0.1381966011250105, 0.5854101966249685, 0.1381966011250105],
							  [0.1381966011250105, 0.1381966011250105, 0.5854101966249685]]

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
				out[e][v, :] = cs[1:] - cs[0]
		return out

	@staticmethod
	def get_ke(mesh, material, nodes_el, jac, volume):
		out_k = lil_array((mesh.n_points, mesh.n_points))
		inv_j = np.linalg.inv(jac)
		w = 1 / 24
		dn = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0], [- 1, 0, 0, 1]])
		# dn = np.array([[-1, 1, 0], [-1, 0, 1], [- 1, 0, 0]])
		b = inv_j.dot(dn)

		k_mat = material['k'] + np.zeros((3, 3))
		dni_k_dnj = b.transpose().dot(k_mat.dot(b))
		# out_k = lil_array(4*w*dni_k_dnj)
		for ii, i in enumerate(nodes_el):
			for jj, j in enumerate(nodes_el):
				# dni_0 = sum(dn[ii] * inv_j[0])
				# dni_1 = sum(dn[ii] * inv_j[1])
				# dni_2 = sum(dn[ii] * inv_j[2])
				# dnj_0 = sum(dn[jj] * inv_j[0])
				# dnj_1 = sum(dn[jj] * inv_j[1])
				# dnj_2 = sum(dn[jj] * inv_j[2])
				# dnm = (dni_0 + dni_1 + dni_2) * (dnj_0 + dnj_1 + dnj_2)
				out_k[i, j] = 4 * w * dni_k_dnj[ii, jj]
		return out_k

	@staticmethod
	def get_me(mesh, material, el, nodes_el, jac, ns, volume):
		out_m = lil_array((mesh.n_points, mesh.n_points))
		det_j = np.linalg.det(jac)
		rhocv = material['rho'] * material['cv']
		integration_points = [[0.1381966, 0.1381966, 0.1381966], [0.58541020, 0.1381966, 0.1381966],
							  [0.1381966, 0.58541020, 0.1381966], [0.1381966, 0.1381966, 0.58541020]]
		w = 1 / 24
		for jj, j in enumerate(nodes_el):
			for ii, i in enumerate(nodes_el):
				temp = []
				for ip in range(len(integration_points)):
					nj = ns[el][jj][ip]
					ni = ns[el][ii][ip]
					temp.append(
						rhocv * ni * nj * w)  # do I need det_j? I think the Jacobian is needed only when you have derivatives
				out_m[i, j] = sum(temp)
		return out_m

	@staticmethod
	def get_neighbours(mesh, nodes_in_element):
		output_variable = lil_array((mesh.n_faces, mesh.n_faces))
		matrix_elements = Mesh.get_nodes_matrix(nodes_in_element)
		for n in range(mesh.n_points):
			elements = np.where(matrix_elements == n)[0]
			combos = list(itertools.combinations(elements, 2))
			for c in combos:
				output_variable[c[0], c[1]] = 1
				output_variable[c[1], c[0]] = 1
		return output_variable

	@staticmethod
	def get_nodes_matrix(nds_els):
		output_variable = np.zeros((len(nds_els), len(nds_els[0])))
		for n in nds_els:
			output_variable[n, :] = nds_els[n]
		return output_variable

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
		return out, np.absolute(sixv)

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
					out_n = np.zeros((int(temp[0]), int(temp[1])))
					out_f = []
					feat = 0
					if temp[2] == 1:
						feat = 1
				else:
					out_n[int(temp[0]), :] = temp[1:4]
					if feat:
						if int(temp[-1]) > 0:
							out_f.append(1)
						else:
							out_f.append(int(temp[-1]))
			return out_n, out_f

	@staticmethod
	def read_properties(file_path):
		out = {}
		with open(file_path) as F:
			for r in F.readlines():
				if '#' not in r:
					temp = r.split('\n')[0].split('->')
					variable = temp[0].split('[')[0].strip()
					unit = temp[0].split('[')[1].split(']')[0].strip()
					has_time, unit = utility.contains_time_unit(unit)
					if has_time:
						value, unit2 = utility.convert_in_hours(temp[1] + unit)
					else:
						value = float(temp[1])
					out[variable] = value
		return out

	@staticmethod
	def scale_cylinder(nds, rds, hgt):
		# Function that scales the cylinder to the appropriate size (rds= radius, hgt= height).
		out = np.zeros_like(nds)

		for i, (x, y, z) in enumerate(nds):
			out[i, 0] = rds * (x / max(nds[:, 0]))
			out[i, 1] = rds * (y / max(nds[:, 1]))
			out[i, 2] = hgt * (z / max(nds[:, 2]))

		return out
