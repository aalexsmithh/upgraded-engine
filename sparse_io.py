import numpy as np
from scipy import sparse, io

def saveArray(arrayX, arrayY, filename):
	assert len(arrayX) == len(arrayY)
	index = len(arrayX)/3
	print len(arrayX), index
	X_1 = filename+"_X_a.mtx"
	X_2 = filename+"_X_b.mtx"
	X_3 = filename+"_X_c.mtx"
	Y_1 = filename+"_Y_a.mtx"
	Y_2 = filename+"_Y_b.mtx"
	Y_3 = filename+"_Y_c.mtx"
	print X_1,X_2,X_3,Y_1,Y_2,Y_3
	io.mmwrite(X_1, sparse.csr_matrix(arrayX[0:index]))
	io.mmwrite(X_2, sparse.csr_matrix(arrayX[index:index*2]))
	io.mmwrite(X_3, sparse.csr_matrix(arrayX[index*2:index*3+1]))
	io.mmwrite(Y_1, sparse.csr_matrix(arrayY[0:index]))
	io.mmwrite(Y_2, sparse.csr_matrix(arrayY[index:index*2]))
	io.mmwrite(Y_3, sparse.csr_matrix(arrayY[index*2:index*3+1]))
	

def openArray(filename):
	X_1 = filename+"_X_a.mtx"
	X_2 = filename+"_X_b.mtx"
	X_3 = filename+"_X_c.mtx"
	Y_1 = filename+"_Y_a.mtx"
	Y_2 = filename+"_Y_b.mtx"
	Y_3 = filename+"_Y_c.mtx"
	X = sparse.vstack((io.mmread(X_1),io.mmread(X_2),io.mmread(X_3)))
	X = np.asarray(X.toarray())
	Y = sparse.hstack((io.mmread(Y_1),io.mmread(Y_2),io.mmread(Y_3)))
	Y = np.asarray(Y.toarray())
	return X, Y.T