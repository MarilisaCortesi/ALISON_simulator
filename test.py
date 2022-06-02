import dill
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
'''
file_in = 'meshes/tetgen_80K_elements/tetgen_80K_elements_OG.pickle'
with open(file_in, 'rb') as F:
	struct = dill.load(F)
[u, s, vh] = scipy.linalg.svd(struct.m.todense())
dill.dump_session('test.pickle')
'''

file = 'test.pickle'
dill.load_session(file)
mm = scipy.sparse.csr_matrix.min(struct.m)
MM = scipy.sparse.csr_matrix.max(struct.m)
struct.m = struct.m*1000
inv_m_1 = scipy.sparse.linalg.inv(struct.m)
mm_inv = scipy.sparse.csr_matrix.min(inv_m_1)
MM_inv = scipy.sparse.csr_matrix.max(inv_m_1)
#c_m = np.linalg.cond(struct.m.todense()) #68957.2
#c_s = np.linalg.cond(np.diag(s))
s[s < 10**-6] = 0
u[u<10**-6] =0
vh[vh<10**-6] =0
el1 = np.dot(np.diag(s**-1), u.transpose())
inv_m = np.dot(vh.transpose(), el1)
print('s')